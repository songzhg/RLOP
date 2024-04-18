#pragma once
#include "rlop/common/torch_utils.h"

namespace rlop {
    class RLBuffer {
    public:
        RLBuffer(
            Int buffer_size, 
            Int num_envs, 
            const std::vector<Int>& observation_sizes, 
            const std::vector<Int>& action_sizes,
            torch::Dtype observation_type = torch::kFloat32,
            torch::Dtype action_type = torch::kFloat32,
            const torch::Device& device = torch::kCPU
        ) :
            buffer_size_(buffer_size), 
            num_envs_(num_envs), 
            observation_sizes_(observation_sizes), 
            action_sizes_(action_sizes),
            observation_type_(observation_type),
            action_type_(action_type), 
            device_(device)
        {}

        ~RLBuffer() = default;

        virtual void Reset() {
            pos_ = 0;
            full_ = false;
        }

        Int Size() const {
            if (full_)
                return buffer_size_;
            return pos_;
        }

        static torch::Tensor SwapAndFlatten(const torch::Tensor& tensor) {
            if (tensor.sizes().size() < 3) 
                return tensor.transpose(0, 1).flatten();
            std::vector<Int> sizes(tensor.sizes().begin()+1, tensor.sizes().end());
            sizes[0] = tensor.sizes()[0] * tensor.sizes()[1];
            return tensor.transpose(0, 1).reshape(sizes);
        }

        Int buffer_size() const {
            return buffer_size_;
        }

        Int num_envs() const {
            return num_envs_;
        }

        Int pos() const {
            return pos_;
        }

        const std::vector<Int>& observation_sizes() const {
            return observation_sizes_;
        }

        const std::vector<Int>& action_sizes() const {
            return action_sizes_;
        }

        const torch::Dtype& observation_type() const {
            return observation_type_;
        }

        const torch::Dtype& action_type() const {
            return action_type_;
        }

        bool full() const {
            return full_;
        }

        const torch::Device& device() const {
            return device_;
        }

    protected:
        Int buffer_size_;
        Int num_envs_;
        Int pos_ = 0;
        std::vector<Int> observation_sizes_;
        std::vector<Int> action_sizes_;
        torch::Dtype observation_type_;
        torch::Dtype action_type_;
        bool full_ = false;
        torch::Device device_;
    };

    class ReplayBuffer : public RLBuffer {
    public:
        struct Batch {
            torch::Tensor observations;
            torch::Tensor actions;
            torch::Tensor next_observations;
            torch::Tensor rewards;
            torch::Tensor dones;

            Batch To(const torch::Device& device) {
                Batch batch;
                batch.observations = observations.to(device);
                batch.actions = actions.to(device);
                batch.next_observations = next_observations.to(device);
                batch.rewards = rewards.to(device);
                batch.dones = dones.to(device);
                return batch;
            }
        };

        ReplayBuffer(
            Int buffer_capacity,
            Int num_envs,
            const std::vector<Int>& observation_sizes, 
            const std::vector<Int>& action_sizes,
            torch::Dtype observation_type = torch::kFloat32,
            torch::Dtype action_type = torch::kFloat32,
            const torch::Device& device = torch::kCPU
        ) :
            RLBuffer(
                std::max((Int)(buffer_capacity / num_envs), Int(1)), 
                num_envs, 
                observation_sizes, 
                action_sizes,
                observation_type,
                action_type, 
                device
            )
        {
            std::vector<Int> observation_buffer_sizes = { buffer_size_, num_envs };
            std::vector<Int> action_buffer_sizes = { buffer_size_, num_envs };
            observation_buffer_sizes.insert(observation_buffer_sizes.end(), observation_sizes_.begin(), observation_sizes_.end());
            action_buffer_sizes.insert(action_buffer_sizes.end(), action_sizes_.begin(), action_sizes_.end()); 
            observations_ = torch::zeros(observation_buffer_sizes, observation_type_).to(device_);
            actions_ = torch::zeros(action_buffer_sizes, action_type_).to(device_);
            next_observations_ = torch::zeros(observation_buffer_sizes, observation_type_).to(device_);
            rewards_ = torch::zeros({ buffer_size_, num_envs }).to(device_);
            dones_ = torch::zeros({ buffer_size_, num_envs }).to(device_);
        }

        virtual ~ReplayBuffer() = default;

        virtual Batch Sample(Int batch_size) {
            Batch batch;
            torch::Tensor batch_indices = torch::randint(0, Size(), {batch_size}).to(device_);
            torch::Tensor env_indices = torch::randint(0, num_envs_, {batch_size}).to(device_);
            batch.observations = observations_.index({batch_indices, env_indices, "..."}).clone();
            batch.next_observations = next_observations_.index({batch_indices, env_indices, "..."}).clone();
            batch.actions = actions_.index({batch_indices, env_indices, "..."}).clone();
            batch.rewards = rewards_.index({batch_indices, env_indices, "..."}).clone();
            batch.dones = dones_.index({batch_indices, env_indices, "..."}).clone();
            return batch;
        }

        virtual void Add(
            const torch::Tensor& observations,
            const torch::Tensor& actions,
            const torch::Tensor& next_observations,
            const torch::Tensor& rewards,
            const torch::Tensor& dones
        ) {
            observations_[pos_].copy_(observations);
            actions_[pos_].copy_(actions);
            next_observations_[pos_].copy_(next_observations);
            rewards_[pos_].copy_(rewards);
            dones_[pos_].copy_(dones);
            pos_ += 1;
            if (pos_ >= buffer_size_) {
                full_ = true;
                pos_ = 0;
            }
        }

        virtual void Load(const std::string& path) {
            torch::serialize::InputArchive archive;
            archive.load_from(path);
            LoadArchive(&archive); 
        }

        virtual void LoadArchive(torch::serialize::InputArchive* archive) {
            archive->read("observations", observations_);
            archive->read("actions", actions_);
            archive->read("next_observations", next_observations_);
            archive->read("rewards", rewards_);
            archive->read("dones", dones_);
            torch::Tensor tensor;
            archive->read("pos", tensor);
            pos_ = tensor.item<Int>();
            tensor = torch::Tensor();
            archive->read("full", tensor);
            full_ = tensor.item<bool>();
            tensor = torch::Tensor();
            archive->read("optimize_memory_usage", tensor);
            observation_sizes_ = observations_.sizes().vec();
            action_sizes_ = actions_.sizes().vec();
            observation_type_ = observations_.scalar_type();
            action_type_ = actions_.scalar_type();
        }

        virtual void Save(const std::string& path) {
            torch::serialize::OutputArchive archive;
            SaveArchive(&archive);
            archive.save_to(path);
        }

        virtual void SaveArchive(torch::serialize::OutputArchive* archive) {
            archive->write("observations", observations_);
            archive->write("actions", actions_);
            archive->write("next_observations", next_observations_);
            archive->write("rewards", rewards_);
            archive->write("dones", dones_);
            archive->write("pos", torch::tensor(pos_));
            archive->write("full", torch::tensor(full_));
        }

        const torch::Tensor& observations() const {
            return observations_;
        }

        const torch::Tensor& actions() const {
            return actions_;
        }

        const torch::Tensor& next_observations() const {
            return next_observations_;
        }

        const torch::Tensor& rewards() const {
            return rewards_;
        }

        const torch::Tensor& dones() const {
            return dones_;
        }

    protected:
        torch::Tensor observations_;
        torch::Tensor actions_;
        torch::Tensor next_observations_;
        torch::Tensor rewards_;
        torch::Tensor dones_;
    };

    class RolloutBuffer : public RLBuffer {
    public:
        struct Batch {
            torch::Tensor observations;
            torch::Tensor actions;
            torch::Tensor values;
            torch::Tensor log_prob;
            torch::Tensor advantages;
            torch::Tensor returns;

            Batch To(const torch::Device& device) {
                Batch batch;
                batch.observations = observations.to(device);
                batch.actions = actions.to(device);
                batch.values = values.to(device);
                batch.log_prob = log_prob.to(device);
                batch.advantages = advantages.to(device);
                batch.returns = returns.to(device);
                return batch;
            }
        };

        RolloutBuffer(
            Int num_steps, 
            Int num_envs, 
            const std::vector<Int>& observation_sizes, 
            const std::vector<Int>& action_sizes,
            torch::Dtype observation_type = torch::kFloat32,
            torch::Dtype action_type = torch::kFloat32,
            const torch::Device& device = torch::kCPU
        ) :
            RLBuffer(
                num_steps, 
                num_envs, 
                observation_sizes, 
                action_sizes,
                observation_type,
                action_type, 
                device
            ),
            observation_buffer_sizes_({ num_steps, num_envs }),
            action_buffer_sizes_({ num_steps, num_envs })
        {
            observation_buffer_sizes_.insert(observation_buffer_sizes_.end(), observation_sizes_.begin(), observation_sizes_.end());
            action_buffer_sizes_.insert(action_buffer_sizes_.end(), action_sizes_.begin(), action_sizes_.end()); 
        }

        virtual ~RolloutBuffer() = default;

        virtual void Reset() override {
            RLBuffer::Reset();
            start_i_ = 0;
            generator_ready_ = false;
            observations_ = torch::zeros(observation_buffer_sizes_, observation_type_).to(device_);
            actions_ = torch::zeros(action_buffer_sizes_, action_type_).to(device_);
            values_ = torch::zeros({ buffer_size_, num_envs_}).to(device_);
            log_probs_ = torch::zeros({ buffer_size_, num_envs_ }).to(device_);
            advantages_ = torch::zeros({ buffer_size_, num_envs_ }).to(device_);
            returns_ = torch::zeros({ buffer_size_, num_envs_ }).to(device_);
            rewards_ = torch::zeros({ buffer_size_, num_envs_ }).to(device_);
            episode_starts_ = torch::zeros({ buffer_size_, num_envs_ }).to(device_);
        }

        virtual Batch Get(Int batch_size) {
            Int num_rollouts = Size() * num_envs_;
            if (!generator_ready_) {
                observations_ = SwapAndFlatten(observations_);
                actions_ = SwapAndFlatten(actions_);
                values_ = SwapAndFlatten(values_);
                log_probs_ = SwapAndFlatten(log_probs_);
                advantages_ = SwapAndFlatten(advantages_);
                returns_ = SwapAndFlatten(returns_);
                generator_ready_ = true;
            }
            if (start_i_ == 0)
                indices_ = torch::randperm(num_rollouts).to(device_);
            Batch batch;
            torch::Tensor indices = indices_.slice(0, start_i_, start_i_ + batch_size);
            batch.observations = observations_.index_select(0, indices); 
            batch.actions = actions_.index_select(0, indices); 
            batch.values = values_.index_select(0, indices); 
            batch.log_prob = log_probs_.index_select(0, indices); 
            batch.advantages = advantages_.index_select(0, indices); 
            batch.returns = returns_.index_select(0, indices); 
            start_i_+=batch_size;
            if (start_i_ >= num_rollouts) 
                start_i_ = 0;
            return batch;
        }

        virtual void Add(
            const torch::Tensor& observations,
            const torch::Tensor& actions,
            const torch::Tensor& values,
            const torch::Tensor& log_prob,
            const torch::Tensor& rewards,
            const torch::Tensor& episode_starts
        ) {
            observations_[pos_].copy_(observations);
            actions_[pos_].copy_(actions);
            values_[pos_].copy_(values);
            log_probs_[pos_].copy_(log_prob);
            rewards_[pos_].copy_(rewards);
            episode_starts_[pos_].copy_(episode_starts);
            pos_ += 1;
            if (pos_ >= buffer_size_) {
                full_ = true;
                pos_ = 0;
            }
        }

        virtual void UpdateGAE(const torch::Tensor& last_values, const torch::Tensor& dones, double gamma, double gae_lambda) {
            torch::Tensor last_gae_lam = torch::zeros(num_envs_).to(device_);
            Int size = Size();
            for (Int i=size-1; i>=0; --i) {
                torch::Tensor next_non_terminal = i == size - 1? 1.0 - dones.to(advantages_.dtype()).to(device_) : 1.0 - episode_starts_[i+1];
                torch::Tensor next_values = i == size - 1? last_values.to(device_) : values_[i+1]; 
                torch::Tensor delta = rewards_[i] + gamma * next_values * next_non_terminal - values_[i];
                last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam;
                advantages_[i].copy_(last_gae_lam);
            }
            returns_ = advantages_ + values_;
        }

        Int start_i() const {
            return start_i_;
        }

        bool generator_ready() const {
            return generator_ready_;
        }

        const torch::Tensor& observations() const {
            return observations_;
        }

        const torch::Tensor& actions() const {
            return actions_;
        }

        const torch::Tensor& values() const {
            return values_;
        }

        const torch::Tensor& log_probs() const {
            return log_probs_;
        }

        const torch::Tensor& advantages() const {
            return advantages_;
        }

        const torch::Tensor& returns() const {
            return returns_;
        }

        const torch::Tensor& rewards() const {
            return rewards_;
        }

        const torch::Tensor& episode_starts() const {
            return episode_starts_;
        }

        const torch::Tensor& indices() const {
            return indices_;
        }

        const std::vector<Int>& observation_buffer_sizes() const {
            return observation_buffer_sizes_;
        }

        const std::vector<Int>& action_buffer_sizes() const {
            return action_buffer_sizes_;
        }
        
    protected:
        Int start_i_ = 0;
        bool generator_ready_ = false;
        torch::Tensor observations_;
        torch::Tensor actions_;
        torch::Tensor values_;
        torch::Tensor log_probs_;
        torch::Tensor advantages_;
        torch::Tensor returns_;
        torch::Tensor rewards_;
        torch::Tensor episode_starts_;
        torch::Tensor indices_;
        std::vector<Int> observation_buffer_sizes_;
        std::vector<Int> action_buffer_sizes_;
    };
}