#pragma once
#include "rlop/common/base_algorithm.h"
#include "rlop/common/platform.h"
#include "rlop/common/torch_utils.h"
#include "rlop/common/utils.h"

namespace rlop {
    // The RL class is as an abstract base class for reinforcement learning algorithms, providing common interfaces
    // for training, managing environments, and performing evaluations.
    class RL : public BaseAlgorithm {
    public:
        // Constructs an RL algorithm instance with a specified output path for logging and a computation device.
        //
        // Parameters:
        //   output_path: Path where training logs and model checkpoints will be saved.
        //   device: The libtorch computation device (e.g., CPU, CUDA GPU).
        RL(const std::string& output_path, const torch::Device& device) :
            output_path_(output_path),
            device_(device)
        {}

        virtual ~RL() = default;

        // Pure virtual function to return the number of environments being managed by this RL instance.
        virtual Int NumEnvs() const = 0;

        // Pure virtual function to reset the environment to its initial state.
        //
        // Returns:
        //   Observations: The initial observations of an episode. 
        virtual torch::Tensor ResetEnv() = 0;

        // Pure virtual function to perform a step in the environment using the provided actions.
        //
        // Parameters:
        //   actions: Actions to take.
        //
        // Returns:
        //   std::array<torch::Tensor, 5>: A tuple containing four elements:
        //     - [0]: Observations - The next observations from the environment after taking the actions.
        //     - [1]: Rewards - The rewards obtained after taking the actions.
        //     - [2]: Terminations - A boolean flag indicating whether agent reaches the terminal state.
        //     - [3]: Truncations - A boolean flag indicating whether the truncation condition outside the scope of
        //            the MDP is satisfied.
        //     - [4]: Final observations - The last observations of an episode.
        virtual std::array<torch::Tensor, 5> Step(const torch::Tensor& actions) = 0;

        // Pure virtual function to collect rollouts from the environment. 
        virtual void CollectRollouts() = 0;

        // Get the policy action from an observation (and optional hidden state). Includes sugar-coating to handle 
        // different observations (e.g. normalizing images).
        //
        // Parameters:
        // observation: the input observation
        //   param state: The last hidden states (can be None, used in recurrent policies)
        //   episode_start: The last masks (can be None, used in recurrent policies) this correspond to beginning of
        //                  episodes, where the hidden states of the RNN must be reset.
        //     
        //   param deterministic: Whether or not to return deterministic actions.
        //
        // Returns: 
        //   std::array<torch::Tensor, 2>: An array containing:
        //     - [0]: Action - The model's actions recommended by the policy for the given observation.
        //     - [1]: State - The next hidden state (used in recurrent policies)
        virtual std::array<torch::Tensor, 2> Predict(const torch::Tensor& observation, bool deterministic = false, const torch::Tensor& state = torch::Tensor(), const torch::Tensor& episode_start = torch::Tensor()) = 0;

        // Pure virtual function to train the model on collected experience.
        virtual void Train() = 0;

        // Resets the algorithm.
        virtual void Reset() override {
            num_iters_ = 0;
            time_steps_ = 0;
            num_updates_ = 0;
            RegisterLogItems();
            if (!output_path_.empty()) {
                std::ofstream out(output_path_ + "_log.txt");
                out << "time_steps";
                for (const auto& pair : log_items_) {
                    out << "\t" << pair.first;
                }
                out << std::endl;
            }
        }

        // Registers loggable items. This function should be overridden to include algorithm-specific metrics.
        virtual void RegisterLogItems() {
            log_items_["num_updates"] = torch::Tensor();
        }

        // Checks if the search should continue.
        virtual bool Proceed() {
            return time_steps_ < max_time_steps_;
        }
        
        // Main learning loop. Runs the algorithm for a specified number of time steps, monitoring and checkpointing as configured.
        virtual void Learn(Int max_time_steps, Int monitor_interval = 0, Int checkpoint_interval = 0) {
            time_steps_ = 0;
            max_time_steps_ = max_time_steps;
            monitor_interval_ = monitor_interval;
            checkpoint_interval_ = checkpoint_interval;
            while (Proceed()) {
                CollectRollouts();
                Train();
                Monitor();
                Checkpoint();
                Update();
            }
        }

        // Monitors the learning progress and logs metrics at specified intervals.
        virtual void Monitor() {
            if (monitor_interval_ <= 0 || num_iters_ % monitor_interval_ != 0)
                return;
            PrintLog(); 
            if (!output_path_.empty())
                SaveLog(output_path_ + "_log.txt");
        }

        // Saves a checkpoint of the model at specified intervals.
        virtual void Checkpoint() {
            if (checkpoint_interval_ <= 0 || num_iters_ % checkpoint_interval_ != 0)
                return; 
            if (!output_path_.empty())
                Save(output_path_ + "_" + rlop::GetDatetime() + "_" + std::to_string(time_steps_) + ".pth");
        }

        virtual void Update() {
            ++num_iters_;
        }

        // Prints the current loggable metrics to the console.
        virtual void PrintLog() const {
            std::cout << std::fixed << std::setw(12) << "time_steps";
            for (const auto& pair : log_items_) {
                std::cout << "\t";
                std::cout << std::fixed << std::setw(12) << pair.first;
            }
            std::cout << std::endl;
            std::cout << std::fixed << std::setw(12) << time_steps_;
            for (const auto& pair : log_items_) {
                std::cout << "\t";
                const auto& dtype = pair.second.scalar_type();
                if (dtype == torch::kDouble || dtype == torch::kFloat64)
                    std::cout << std::fixed << std::setw(12) << pair.second.cpu().item<double>(); 
                else if (dtype == torch::kFloat || dtype == torch::kFloat32) 
                    std::cout << std::fixed << std::setw(12) << pair.second.cpu().item<float>();
                else if (dtype == torch::kInt64) 
                    std::cout << std::fixed << std::setw(12) << pair.second.cpu().item<Int>();
                else if (dtype == torch::kBool) 
                    std::cout << std::fixed << std::setw(12) << pair.second.cpu().item<bool>();
            }
            std::cout << std::endl;
        }

        // Saves the current loggable metrics to a file.
        virtual void SaveLog(const std::string& path) {
            if (path.empty())
                return;
            std::ofstream out(path, std::ios::app);
            out << time_steps_;
            for (const auto& pair : log_items_) {
                out << "\t";
                const auto& dtype = pair.second.scalar_type();
                if (dtype == torch::kDouble || dtype == torch::kFloat64)
                    out << pair.second.cpu().item<double>(); 
                else if (dtype == torch::kFloat || dtype == torch::kFloat32) 
                    out << pair.second.cpu().item<float>();
                else if (dtype == torch::kInt64) 
                    out << pair.second.cpu().item<Int>();
                else if (dtype == torch::kBool) 
                    out << pair.second.cpu().item<bool>();
            }
            out << std::endl;
        }

        // Loads model and algorithm state from a file.
        virtual void Load(const std::string& path, const std::unordered_set<std::string>& names = {"all"}) {
            torch::serialize::InputArchive archive;
            archive.load_from(path);
            LoadArchive(&archive, names); 
        }
        
        // Saves model and algorithm state to a file.
        virtual void Save(const std::string& path, const std::unordered_set<std::string>& names = {"all"}) {
            torch::serialize::OutputArchive archive;
            SaveArchive(&archive, names);
            archive.save_to(path);
        }

        // Load algorithm-specific components from an archive. To be implemented by derived classes.
        virtual void LoadArchive(torch::serialize::InputArchive* archive, const std::unordered_set<std::string>& names) {}

        // Save algorithm-specific components to an archive. To be implemented by derived classes.
        virtual void SaveArchive(torch::serialize::OutputArchive* archive, const std::unordered_set<std::string>& names) {}

    protected:
        Int num_iters_ = 0;
        Int time_steps_ = 0;
        Int max_time_steps_ = 0;
        Int num_updates_ = 0;
        Int monitor_interval_ = 0;
        Int checkpoint_interval_ = 0;
        std::string output_path_;
        std::unordered_map<std::string, torch::Tensor> log_items_;
        torch::Device device_;
    };
}