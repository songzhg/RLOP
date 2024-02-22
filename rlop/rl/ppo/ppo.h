#pragma once
#include "policy.h"
#include "rlop/rl/rl.h"
#include "rlop/rl/buffers.h"

namespace rlop {
    class PPO : public RL {
    public:
        PPO(
            Int batch_size = 64,
            Int num_epochs = 10,
            double lr = 1e-6,
            double gamma = 0.99,
            double clip_range = 0.2,
            double clip_range_vf = 0,
            bool normalize_advantage = true,
            double ent_coef = 0,
            double vf_coef = 0.5,
            double gae_lambda = 0.95,
            double max_grad_norm = 10,
            double target_kl = 0.1,
            const std::string& output_path = "",
            const torch::Device& device = torch::kCPU
        ) :
            RL(output_path, device),
            batch_size_(batch_size),
            num_epochs_(num_epochs),
            lr_(lr),
            gamma_(gamma),
            clip_range_(clip_range),
            clip_range_vf_(clip_range_vf),
            normalize_advantage_(normalize_advantage),
            ent_coef_(ent_coef),
            vf_coef_(vf_coef),
            gae_lambda_(gae_lambda),
            max_grad_norm_(max_grad_norm),
            target_kl_(target_kl)
        {}

        virtual ~PPO() = default;

        // Pure virtual function to create and return a unique pointer to a RolloutBuffer object.
        virtual std::unique_ptr<RolloutBuffer> MakeRolloutBuffer() const = 0; 

        // Pure virtual function to create and return a unique pointer to a PPOPolicy object.
        virtual std::unique_ptr<PPOPolicy> MakePPOPolicy() const = 0;
       
        virtual void Reset() override {
            RL::Reset();
            rollout_buffer_ = MakeRolloutBuffer();
            policy_ = MakePPOPolicy();
            policy_->to(device_);
            optimizer_ = MakeOptimizer();
            observation_ = ResetEnv();
            episode_start_ = torch::ones({ rollout_buffer_->num_envs() }, torch::kBool);
        }
    
        virtual void RegisterLogItems() override {
            RL::RegisterLogItems();
            log_items_["ratio"] = torch::Tensor();
            log_items_["policy_loss"] = torch::Tensor();
            log_items_["value_loss"] = torch::Tensor();
            log_items_["entropy_loss"] = torch::Tensor();
            log_items_["loss"] = torch::Tensor();
            log_items_["approx_kl"] = torch::Tensor();
            log_items_["variance"] = torch::Tensor();
            log_items_["return"] = torch::Tensor();
        }

        virtual void CollectRollouts() override {
            policy_->eval(); 
            torch::NoGradGuard no_grad;
            rollout_buffer_->Reset();
            while (!rollout_buffer_->full()) {
                auto [ action, value, log_prob ] = policy_->Forward(observation_.to(device_));
                auto [ next_observation, reward, done ] = Step(action);
                rollout_buffer_->Add(observation_, action, value, log_prob, reward, episode_start_);
                observation_ = next_observation;
                episode_start_ = done;
                time_steps_ += NumEnvs();
            }
            torch::Tensor value = policy_->PredictValue(observation_.to(device_));
            rollout_buffer_->UpdateGAE(value, episode_start_, gamma_, gae_lambda_);
        }

        virtual std::array<torch::Tensor, 2> Predict(const torch::Tensor& observation, bool deterministic = false, const torch::Tensor& state = torch::Tensor(), const torch::Tensor& episode_start = torch::Tensor()) {
            return policy_->Predict(observation.to(device_), deterministic, state, episode_start);
        }
    
        virtual void Train() override {
            Int num_steps = rollout_buffer_->Size() * rollout_buffer_->num_envs() / batch_size_;
            std::vector<torch::Tensor> ratio_list;
            std::vector<torch::Tensor> policy_loss_list;
            std::vector<torch::Tensor> value_loss_list;
            std::vector<torch::Tensor> entropy_loss_list;
            std::vector<torch::Tensor> loss_list;
            std::vector<torch::Tensor> approx_kl_list;
            Int reserved_size = num_epochs_*(num_steps + 1);
            ratio_list.reserve(reserved_size);
            policy_loss_list.reserve(reserved_size);
            value_loss_list.reserve(reserved_size);
            entropy_loss_list.reserve(reserved_size);
            loss_list.reserve(reserved_size);
            approx_kl_list.reserve(reserved_size);
            bool continue_training = true;
            policy_->train();
            for (Int epoch=0; epoch<num_epochs_; ++epoch) {
                for (Int step =0; step<=num_steps; ++step) {
                    auto batch = rollout_buffer_->Get(batch_size_).to(device_);
                    auto [ value, log_prob, entropy ] = policy_->EvaluateAction(batch.observation, batch.action);
                    if (normalize_advantage_ && batch.advantage.sizes()[0] > 1)
                        batch.advantage = (batch.advantage - batch.advantage.mean()) / (batch.advantage.std() + 1e-8);
                    torch::Tensor ratio = torch::exp(log_prob - batch.log_prob);
                    torch::Tensor policy_loss_1 = batch.advantage * ratio;
                    torch::Tensor policy_loss_2 = batch.advantage * torch::clamp(ratio, 1 - clip_range_, 1 + clip_range_);
                    torch::Tensor policy_loss = -torch::min(policy_loss_1, policy_loss_2).mean();
                    torch::Tensor pred_value;
                    if (clip_range_vf_ > 0)
                        pred_value = batch.value + torch::clamp(value - batch.value, -clip_range_vf_, clip_range_vf_);
                    else
                        pred_value = value; 
                    torch::Tensor value_loss = torch::mse_loss(batch.ret, pred_value);
                    torch::Tensor entropy_loss;
                    if (entropy)
                        entropy_loss = -torch::mean(*entropy);
                    else
                        entropy_loss = -torch::mean(-log_prob);
                    torch::Tensor loss = policy_loss + vf_coef_ * value_loss + ent_coef_ * entropy_loss;
                    float approx_kl_div;
                    {
                        torch::NoGradGuard no_grad;
                        approx_kl_div = torch_utils::ComputeApproxKL(log_prob, batch.log_prob).to(torch::kCPU).item<float>();
                    }
                    if (target_kl_ > 0 && approx_kl_div > 1.5 * target_kl_) {
                        std::cout << "Early stopping at epoch " << epoch << " due to reaching max kl: {approx_kl_div: " << approx_kl_div << "}"  << std::endl;
                        continue_training = false;
                        break;
                    }
                    optimizer_->zero_grad();
                    loss.backward();
                    torch::nn::utils::clip_grad_norm_(policy_.get()->parameters(), max_grad_norm_);
                    optimizer_->step();
                    ratio_list.push_back(std::move(ratio.detach())); 
                    policy_loss_list.push_back(std::move(policy_loss.detach())); 
                    value_loss_list.push_back(std::move(value_loss.detach())); 
                    entropy_loss_list.push_back(std::move(entropy_loss.detach())); 
                    approx_kl_list.push_back(torch::tensor(approx_kl_div));
                    loss_list.push_back(std::move(loss.detach())); 
                }
                ++num_updates_;
                if (!continue_training)
                    break;
            }
            if (!log_items_.empty()) {
                auto it = log_items_.find("num_updates");
                if (it != log_items_.end()) 
                    it->second = torch::tensor(num_updates_);
                it = log_items_.find("ratio");
                if (it != log_items_.end()) 
                    it->second = torch::stack(ratio_list).mean();
                it = log_items_.find("policy_loss");
                if (it != log_items_.end()) 
                    it->second  = torch::stack(policy_loss_list).mean();
                it = log_items_.find("value_loss");
                if (it != log_items_.end()) 
                    it->second  = torch::stack(value_loss_list).mean();
                it = log_items_.find("entropy_loss");
                if (it != log_items_.end()) 
                    it->second  = torch::stack(entropy_loss_list).mean();
                it = log_items_.find("loss");
                if (it != log_items_.end()) 
                    it->second  = torch::stack(loss_list).mean();
                it = log_items_.find("approx_kl");
                if (it != log_items_.end()) 
                    it->second  = torch::stack(approx_kl_list).mean();
                it = log_items_.find("variance");
                if (it != log_items_.end()) 
                    it->second  = torch_utils::ExplainedVariance(rollout_buffer_->values().flatten(), rollout_buffer_->returns().flatten());
                it = log_items_.find("return");
                if (it != log_items_.end()) 
                    it->second = rollout_buffer_->returns().mean();
            }
        }

        virtual void LoadArchive(torch::serialize::InputArchive* archive, const std::unordered_set<std::string>& names) override {
            if (names.count("all") || names.count("actor_critic_net")) {
                torch::serialize::InputArchive net_archive;
                if (archive->try_read("actor_critic_net", net_archive)) {
                    archive->read("actor_critic_net", net_archive);
                    policy_->load(net_archive);
                }
            }
            if (names.count("all") || names.count("optimizer")) {
                torch::serialize::InputArchive optimizer_archive;
                if (archive->try_read("optimizer", optimizer_archive)) {
                    archive->read("optimizer", optimizer_archive);
                    optimizer_->load(optimizer_archive);
                }
            }
            if (names.count("all") || names.count("hparams")) {
                torch::Tensor tensor;
                if (archive->try_read("batch_size", tensor))
                    batch_size_ = tensor.item<Int>();
                tensor = torch::Tensor();
                if (archive->try_read("num_epochs", tensor))
                    num_epochs_ = tensor.item<Int>();
                tensor = torch::Tensor();
                if (archive->try_read("lr", tensor))
                    lr_ = tensor.item<double>();
                tensor = torch::Tensor();
                if (archive->try_read("gamma", tensor))
                    gamma_ = tensor.item<double>();
                tensor = torch::Tensor();
                if (archive->try_read("clip_range", tensor))
                    clip_range_ = tensor.item<double>();
                tensor = torch::Tensor();
                if (archive->try_read("clip_range_vf", tensor))
                    clip_range_vf_ = tensor.item<double>();
                tensor = torch::Tensor();
                if (archive->try_read("normalize_advantage", tensor))
                    normalize_advantage_ = tensor.item<bool>();
                tensor = torch::Tensor();
                if (archive->try_read("ent_coef", tensor))
                    ent_coef_ = tensor.item<double>();
                tensor = torch::Tensor();
                if (archive->try_read("vf_coef", tensor))
                    vf_coef_ = tensor.item<double>();
                tensor = torch::Tensor();
                if (archive->try_read("gae_lambda", tensor))
                    gae_lambda_ = tensor.item<double>();
                tensor = torch::Tensor();
                if (archive->try_read("max_grad_norm", tensor))
                    max_grad_norm_ = tensor.item<double>();
                tensor = torch::Tensor();
                if (archive->try_read("target_kl", tensor))
                    target_kl_ = tensor.item<double>();
            }
        }

        virtual void SaveArchive(torch::serialize::OutputArchive* archive, const std::unordered_set<std::string>& names) override {
            if (names.count("all") || names.count("actor_critic_net")) {
                torch::serialize::OutputArchive net_archive;
                policy_->save(net_archive);
                archive->write("actor_critic_net", net_archive);
            }
            if (names.count("all") || names.count("optimizer")) {
                torch::serialize::OutputArchive optimizer_archive;
                optimizer_->save(optimizer_archive);
                archive->write("optimizer", optimizer_archive);
            }
            if (names.count("all") || names.count("hparams")) {
                archive->write("batch_size", torch::tensor(batch_size_));
                archive->write("num_epochs", torch::tensor(num_epochs_));
                archive->write("lr", torch::tensor(lr_));
                archive->write("gamma", torch::tensor(gamma_));
                archive->write("clip_range", torch::tensor(clip_range_));
                archive->write("clip_range_vf", torch::tensor(clip_range_vf_));
                archive->write("normalize_advantage", torch::tensor(normalize_advantage_));
                archive->write("ent_coef", torch::tensor(ent_coef_));
                archive->write("vf_coef", torch::tensor(vf_coef_));
                archive->write("gae_lambda", torch::tensor(gae_lambda_));
                archive->write("max_grad_norm", torch::tensor(max_grad_norm_));
                archive->write("target_kl", torch::tensor(target_kl_));
            }
        }

        virtual std::unique_ptr<torch::optim::Optimizer> MakeOptimizer() const {
            return std::make_unique<torch::optim::Adam>(policy_.get()->parameters(), torch::optim::AdamOptions(lr_));
        }

        const std::unique_ptr<RolloutBuffer>& rollout_buffer() const {
            return rollout_buffer_;
        }

        const std::unique_ptr<PPOPolicy>& policy() const {
            return policy_;
        }

        const std::unique_ptr<torch::optim::Optimizer>& optimizer() const {
            return optimizer_;
        }

    protected:
        Int batch_size_;
        Int num_epochs_;
        double lr_;
        double gamma_;
        double clip_range_;
        double clip_range_vf_;
        bool normalize_advantage_;
        double ent_coef_;
        double vf_coef_;
        double gae_lambda_;
        double max_grad_norm_;
        double target_kl_;
        std::unique_ptr<RolloutBuffer> rollout_buffer_ = nullptr;
        std::unique_ptr<PPOPolicy> policy_ = nullptr;
        std::unique_ptr<torch::optim::Optimizer> optimizer_ = nullptr;
        torch::Tensor observation_;
        torch::Tensor episode_start_;
    };
}