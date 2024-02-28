#pragma once
#include "policy.h"
#include "rlop/rl/rl.h"
#include "rlop/rl/buffers.h"

namespace rlop {
    // The DQN class inherits from the RL base class and implements the Deep Q-Network algorithm, a value-based 
    // method for reinforcement learning that uses deep neural networks to approximate Q-values. This 
    // implementation references the DQN implementation of Stable Baselines3.
    // Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    class DQN : public RL {
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
            RL(output_path, device),
            learning_starts_(learning_starts), 
            batch_size_(batch_size), 
            lr_(lr),
            tau_(tau),
            gamma_(gamma),
            eps_(eps),
            max_grad_norm_(max_grad_norm),
            train_freq_(train_freq),
            gradient_steps_(gradient_steps),
            target_update_interval_(target_update_interval)
        {}

        virtual ~DQN() = default;

        // Factory method to create and return a unique pointer to a ReplayBuffer object.
        virtual std::unique_ptr<ReplayBuffer> MakeReplayBuffer() const = 0;

        // Factory method to create and return a unique pointer to a QNet object.
        virtual std::unique_ptr<QNet> MakeQNet() const = 0;

        // Pure virtual function to sample an action from the action space.
        //
        // Returns:
        //   torch::Tensor: A tensor representing the selected actions.
        virtual torch::Tensor SampleAction() = 0;

        virtual void Reset() override {
            RL::Reset();
            replay_buffer_ = MakeReplayBuffer();
            q_net_ = MakeQNet();
            target_q_net_ = MakeQNet();
            q_net_->to(device_);
            target_q_net_->to(device_);
            torch_utils::CopyStateDict(*q_net_, target_q_net_.get());
            target_q_net_->eval();
            optimizer_ = MakeOptimizer();
            last_observation_ = ResetEnv();
        }

        // Factory method to create an optimizer for the Q-network. By default, uses the RMSprop optimizer.
        virtual std::unique_ptr<torch::optim::Optimizer> MakeOptimizer() const {
            return std::make_unique<torch::optim::RMSprop>(q_net_.get()->parameters(), torch::optim::RMSpropOptions(lr_));
        }

        virtual void RegisterLogItems() override {
            RL::RegisterLogItems();
            log_items_["q_value"] = torch::Tensor();
            log_items_["loss"] = torch::Tensor();
            log_items_["reward"] = torch::Tensor();
            log_items_["eps"] = torch::Tensor();
        }

        virtual void StoreTransition(
            const torch::Tensor& action, 
            const torch::Tensor& next_observation, 
            const torch::Tensor& reward, 
            const torch::Tensor& done,
            const torch::Tensor& terminal_observation
        ) {
            torch::Tensor new_observation = next_observation;
            if (terminal_observation.defined()) {
                for (Int i=0; i<replay_buffer_->num_envs(); ++i) {
                    if (done[i].item<bool>() && !torch::isnan(terminal_observation[i]).any().item<bool>()) 
                        new_observation[i] = terminal_observation[i];
                }
            }
            replay_buffer_->Add(last_observation_, action, new_observation, reward, done); 
        }

        virtual void CollectRollouts() override {
            q_net_->eval();
            torch::NoGradGuard no_grad;
            if (num_iters_ == 0) {
                for (Int step = 0; step < learning_starts_; ++step) {
                    torch::Tensor action = SampleAction();
                    auto [next_observation, reward, terminated, truncated, terminal_observation] = Step(action);
                    torch::Tensor done = torch::logical_or(terminated, truncated).to(torch::get_default_dtype());
                    StoreTransition(action, next_observation, reward, done, terminal_observation);
                    last_observation_ = next_observation;
                }
            }
            for (Int step = 0; step < train_freq_; ++step) {
                torch::Tensor action;
                if (torch::rand({1}, torch::kFloat64).item<double>() < eps_)
                    action = SampleAction();
                else 
                    action = q_net_->PredictAction(last_observation_.to(device_));
                auto [next_observation, reward, terminated, truncated, terminal_observation] = Step(action);
                torch::Tensor done = torch::logical_or(terminated, truncated).to(torch::get_default_dtype());
                StoreTransition(action, next_observation, reward, done, terminal_observation);
                last_observation_ = next_observation;
                time_steps_ += NumEnvs();
            }
        }

        virtual std::array<torch::Tensor, 2> Predict(const torch::Tensor& observation, bool deterministic = false, const torch::Tensor& state = torch::Tensor(), const torch::Tensor& episode_start = torch::Tensor()) {
            return q_net_->Predict(observation.to(device_), deterministic, state, episode_start);
        }

        virtual void Train() override {
            q_net_->train();
            std::vector<torch::Tensor> q_value_list;
            std::vector<torch::Tensor> loss_list;
            std::vector<torch::Tensor> reward_list;
            q_value_list.reserve(gradient_steps_);
            loss_list.reserve(gradient_steps_);
            reward_list.reserve(gradient_steps_);
            for (Int step=0; step<gradient_steps_; ++step) {
                auto batch = replay_buffer_->Sample(batch_size_).to(device_);
                torch::Tensor target_q_value;
                {
                    torch::NoGradGuard no_grad;
                    torch::Tensor next_q_values = target_q_net_->Forward(batch.next_observation);
                    torch::Tensor max_next_q_value = std::get<0>(torch::max(next_q_values, 1));
                    target_q_value = batch.reward + (1 - batch.done) * gamma_ * max_next_q_value;
                }
                torch::Tensor q_values = q_net_->Forward(batch.observation);
                torch::Tensor q_value = torch::gather(q_values, 1, batch.action.reshape({-1, 1})).squeeze(-1);
                torch::Tensor loss = torch::smooth_l1_loss(q_value, target_q_value);
                optimizer_->zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(q_net_.get()->parameters(), max_grad_norm_);
                optimizer_->step();
                ++num_updates_;
                q_value_list.push_back(std::move(q_values.detach()));
                loss_list.push_back(std::move(loss.detach()));
                reward_list.push_back(std::move(batch.reward.detach()));
            }
            if (!log_items_.empty()) {
                auto it = log_items_.find("num_updates");
                if (it != log_items_.end())
                it->second = torch::tensor(num_updates_);
                it = log_items_.find("q_value");
                if (it != log_items_.end()) 
                    it->second  = torch::stack(q_value_list).mean();
                it = log_items_.find("loss");
                if (it != log_items_.end()) 
                    it->second = torch::stack(loss_list).mean();
                it = log_items_.find("reward");
                if (it != log_items_.end()) 
                    it->second = torch::stack(reward_list).mean();
                it = log_items_.find("eps");
                if (it != log_items_.end()) 
                    it->second = torch::tensor(eps_);
            }
        }

        virtual void Update() {
            RL::Update();
            if (num_iters_ % std::max(target_update_interval_ / NumEnvs(), Int(1)) == 0) {
                auto [ params_names, params ] = torch_utils::GetParameters(*q_net_);
                auto [ target_params_names, target_params ] = torch_utils::GetParameters(*target_q_net_);
                auto [ buffer_names, buffers ] = torch_utils::GetBuffers(*q_net_);
                auto [ target_buffer_names, target_buffers ] = torch_utils::GetBuffers(*target_q_net_);
                torch_utils::PolyakUpdate(params, target_params, tau_);
                torch_utils::PolyakUpdate(buffers, target_buffers, 1.0);
            }
        }

        virtual void LoadArchive(torch::serialize::InputArchive* archive, const std::unordered_set<std::string>& names) override {
            if (names.count("all") || names.count("q_net")) {
                torch::serialize::InputArchive q_net_archive;
                if (archive->try_read("q_net", q_net_archive)) {
                    archive->read("q_net", q_net_archive);
                    q_net_->load(q_net_archive);
                }
            }
            if (names.count("all") || names.count("target_net")) {
                torch::serialize::InputArchive target_q_net_archive;
                if (archive->try_read("target_net", target_q_net_archive)) {
                    archive->read("target_net", target_q_net_archive);
                    target_q_net_->load(target_q_net_archive);
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
                q_net_->save(net_archive);
                archive->write("q_net", net_archive);
            }
            if (names.count("all") || names.count("target_net")) {
                torch::serialize::OutputArchive net_archive;
                target_q_net_->save(net_archive);
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
       
        const std::unique_ptr<ReplayBuffer>& replay_buffer() const {
            return replay_buffer_;
        }

        const std::unique_ptr<QNet>& q_net() const {
            return q_net_;
        }

        const std::unique_ptr<QNet>& target_q_net() const {
            return target_q_net_;
        }

        const std::unique_ptr<torch::optim::Optimizer>& optimizer() const {
            return optimizer_;
        }

    protected:
        Int learning_starts_;
        Int batch_size_;
        double lr_;
        double tau_;
        double gamma_;
        double eps_;
        double max_grad_norm_;
        Int train_freq_;
        Int gradient_steps_;
        Int target_update_interval_;
        std::unique_ptr<ReplayBuffer> replay_buffer_ = nullptr;
        std::unique_ptr<QNet> q_net_ = nullptr;
        std::unique_ptr<QNet> target_q_net_ = nullptr;
        std::unique_ptr<torch::optim::Optimizer> optimizer_ = nullptr;
        torch::Tensor last_observation_;
    };
}