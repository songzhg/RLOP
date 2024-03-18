#pragma once
#include "rlop/rl/off_policy_rl.h"
#include "policy.h"

namespace rlop {
    // The DQN class inherits from the RL base class and implements the Deep Q-Network algorithm, a value-based 
    // method for reinforcement learning that uses deep neural networks to approximate Q-values. This 
    // implementation references the DQN implementation of Stable Baselines3.
    // Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    class DQN : public OffPolicyRL {
    public:
        DQN(
            Int learning_starts = 100, // Number of steps to observe before starting training.
            Int batch_size = 32, // Size of batches taken from the replay buffer for training.
            double lr = 1e-4, // Learning rate for the RMSprop optimizer.
            double tau = 1.0, // Coefficient for soft update of the target network.
            double gamma = 0.99, // Discount factor for future rewards.
            double eps = 0.1, // Epsilon value for epsilon-greedy action selection.
            double max_grad_norm = 10, // Maximum gradient norm for gradient clipping.
            Int train_freq = 4, // Number of environment steps between each training step.
            Int gradient_steps = 1, // Number of gradient steps per training step.
            Int target_update_interval = 1e4, // Number of steps between updates to the target network.
            const std::string& output_path = "",
            const torch::Device& device = torch::kCPU
        ) :
            batch_size_(batch_size), 
            lr_(lr),
            tau_(tau),
            gamma_(gamma),
            eps_(eps),
            max_grad_norm_(max_grad_norm),
            gradient_steps_(gradient_steps),
            target_update_interval_(target_update_interval),
            OffPolicyRL(learning_starts, train_freq, output_path, device)
        {}

        virtual ~DQN() = default;

        virtual void Reset() override {
            OffPolicyRL::Reset();
            optimizer_ = MakeOptimizer();
            parameters_ = torch_utils::GetParameters(*policy()->q_net()).second;
            target_parameters_ = torch_utils::GetParameters(*policy()->q_net_target()).second;
            buffers_ = torch_utils::GetBuffersByName(*policy()->q_net(), { "running_" });
            target_buffers_ = torch_utils::GetBuffersByName(*policy()->q_net_target(), { "running_" });
            num_calls_ = 0;
        }

        // Factory method to create an optimizer for the Q-network. By default, uses the RMSprop optimizer.
        virtual std::unique_ptr<torch::optim::Optimizer> MakeOptimizer() const {
            return std::make_unique<torch::optim::RMSprop>(policy()->q_net()->parameters(), torch::optim::RMSpropOptions(lr_));
        }

        virtual void RegisterLogItems() override {
            OffPolicyRL::RegisterLogItems();
            log_items_["q_value"] = torch::Tensor();
            log_items_["loss"] = torch::Tensor();
            log_items_["reward"] = torch::Tensor();
            log_items_["eps"] = torch::Tensor();
        }

        virtual std::array<torch::Tensor, 2> Predict(const torch::Tensor& observation, bool deterministic = false, const torch::Tensor& state = torch::Tensor(), const torch::Tensor& episode_start = torch::Tensor()) override {
            if (!deterministic && torch::rand({1}, torch::kFloat64).item<double>() < eps_)
                return { SampleActions(), torch::Tensor() };
            else   
                return policy_->Predict(observation, deterministic, state, episode_start);
        }

        virtual void Train() override {
            if (time_steps_ <= learning_starts_)
                return;
            policy_->SetTrainingMode(true);
            std::vector<double> q_value_list;
            std::vector<double> loss_list;
            std::vector<double> reward_list;
            q_value_list.reserve(gradient_steps_);
            loss_list.reserve(gradient_steps_);
            reward_list.reserve(gradient_steps_);
            for (Int step=0; step<gradient_steps_; ++step) {
                auto batch = replay_buffer_->Sample(batch_size_).To(device_);
                torch::Tensor target_q_value;
                {
                    torch::NoGradGuard no_grad;
                    torch::Tensor next_q_values = policy()->q_net_target()->PredictQValues(batch.next_observations);
                    torch::Tensor max_next_q_value = std::get<0>(torch::max(next_q_values, 1));
                    target_q_value = batch.rewards + (1.0 - batch.dones) * gamma_ * max_next_q_value;
                }
                torch::Tensor q_values = policy()->q_net()->PredictQValues(batch.observations);
                torch::Tensor q_value = torch::gather(q_values, 1, batch.actions.reshape({-1, 1})).flatten();
                torch::Tensor loss = torch::smooth_l1_loss(q_value, target_q_value);
                optimizer_->zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(policy()->q_net()->parameters(), max_grad_norm_);
                optimizer_->step();
                ++num_updates_;
                q_value_list.push_back(q_values.mean().item<double>());
                loss_list.push_back(loss.item<double>());
                reward_list.push_back(batch.rewards.mean().item<double>());
            }
            if (!log_items_.empty()) {
                auto it = log_items_.find("num_updates");
                if (it != log_items_.end())
                it->second = torch::tensor(num_updates_);
                it = log_items_.find("q_value");
                if (it != log_items_.end()) 
                    it->second  = torch::tensor(q_value_list).mean();
                it = log_items_.find("loss");
                if (it != log_items_.end()) 
                    it->second = torch::tensor(loss_list).mean();
                it = log_items_.find("reward");
                if (it != log_items_.end()) 
                    it->second = torch::tensor(reward_list).mean();
                it = log_items_.find("eps");
                if (it != log_items_.end()) 
                    it->second = torch::tensor(eps_);
            }
        }

        virtual void OnCollectRolloutStep() override {
            ++num_calls_;
            if (num_calls_ % std::max(target_update_interval_ / replay_buffer_->num_envs(), Int(1)) == 0) {
                torch_utils::PolyakUpdate(parameters_, target_parameters_, tau_);
                torch_utils::PolyakUpdate(buffers_, target_buffers_, 1.0);
            }
        }

        virtual void LoadArchive(torch::serialize::InputArchive* archive, const std::unordered_set<std::string>& names) override {
            if (names.count("all") || names.count("q_net")) {
                torch::serialize::InputArchive q_net_archive;
                if (archive->try_read("q_net", q_net_archive))
                    policy()->q_net()->load(q_net_archive);
            }
            if (names.count("all") || names.count("target_net")) {
                torch::serialize::InputArchive target_q_net_archive;
                if (archive->try_read("target_net", target_q_net_archive))
                    policy()->q_net_target()->load(target_q_net_archive);
            }
            if (names.count("all") || names.count("optimizer")) {
                torch::serialize::InputArchive optimizer_archive;
                if (archive->try_read("optimizer", optimizer_archive))
                    optimizer_->load(optimizer_archive);
            }
            if (names.count("all") || names.count("hparams")) {
                torch::Tensor tensor;
                if (archive->try_read("learning_starts", tensor))
                    learning_starts_ = tensor.item<Int>();
                tensor = torch::Tensor();
                if (archive->try_read("batch_size", tensor))
                    batch_size_ = tensor.item<Int>();
                tensor = torch::Tensor();
                if (archive->try_read("lr", tensor))
                    lr_ = tensor.item<double>();
                tensor = torch::Tensor();
                if (archive->try_read("tau", tensor))
                    tau_ = tensor.item<double>();
                tensor = torch::Tensor();
                if (archive->try_read("gamma", tensor))
                    gamma_ = tensor.item<double>();
                tensor = torch::Tensor();
                if (archive->try_read("epsilon", tensor))
                    eps_ = tensor.item<double>();
                tensor = torch::Tensor();
                if (archive->try_read("max_grad_norm", tensor))
                    max_grad_norm_ = tensor.item<double>();
                tensor = torch::Tensor();
                if (archive->try_read("train_freq", tensor))
                    train_freq_ = tensor.item<Int>();
                tensor = torch::Tensor();
                if (archive->try_read("gradient_steps", tensor))
                    gradient_steps_ = tensor.item<Int>();
                tensor = torch::Tensor();
                if (archive->try_read("target_update_interval", tensor))
                    target_update_interval_ = tensor.item<Int>();
            }
        }

        virtual void SaveArchive(torch::serialize::OutputArchive* archive, const std::unordered_set<std::string>& names) override {
            if (names.count("all") || names.count("q_net")) {
                torch::serialize::OutputArchive net_archive;
                policy()->q_net()->save(net_archive);
                archive->write("q_net", net_archive);
            }
            if (names.count("all") || names.count("target_net")) {
                torch::serialize::OutputArchive net_archive;
                policy()->q_net_target()->save(net_archive);
                archive->write("target_net", net_archive);
            }
            if (names.count("all") || names.count("optimizer")) {
                torch::serialize::OutputArchive optimizer_archive;
                optimizer_->save(optimizer_archive);
                archive->write("optimizer", optimizer_archive);
            }
            if (names.count("all") || names.count("hparams")) {
                archive->write("learning_starts", torch::tensor(learning_starts_));
                archive->write("batch_size", torch::tensor(batch_size_));
                archive->write("lr", torch::tensor(lr_));
                archive->write("tau", torch::tensor(tau_));
                archive->write("gamma", torch::tensor(gamma_));
                archive->write("epsilon", torch::tensor(eps_));
                archive->write("max_grad_norm", torch::tensor(max_grad_norm_));
                archive->write("train_freq", torch::tensor(train_freq_));
                archive->write("gradient_steps", torch::tensor(gradient_steps_));
                archive->write("target_update_interval", torch::tensor(target_update_interval_));
            }
        }
       
        std::shared_ptr<torch::optim::Optimizer> optimizer() const {
            return optimizer_;
        }

        std::shared_ptr<DQNPolicy> policy() const {
            return std::static_pointer_cast<DQNPolicy>(policy_);
        } 

    protected:
        Int batch_size_;
        double lr_;
        double tau_;
        double gamma_;
        double eps_;
        double max_grad_norm_;
        Int gradient_steps_;
        Int target_update_interval_;
        Int num_calls_;
        std::shared_ptr<torch::optim::Optimizer> optimizer_ = nullptr;
        std::vector<torch::Tensor> parameters_;
        std::vector<torch::Tensor> target_parameters_;
        std::vector<torch::Tensor> buffers_;
        std::vector<torch::Tensor> target_buffers_;
    };
}