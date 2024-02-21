#pragma once
#include "rlop/common/base_algorithm.h"
#include "rlop/common/platform.h"
#include "rlop/common/torch_utils.h"

namespace rlop {
    class RL : public BaseAlgorithm {
    public:
        RL(const std::string& output_path, const torch::Device& device) : 
            output_path_(output_path),
            device_(device)
        {}

        virtual ~RL() = default;

        virtual Int NumEnvs() const = 0;

        virtual torch::Tensor ResetEnv() = 0;

        virtual std::array<torch::Tensor, 3> Step(const torch::Tensor& action) = 0;

        virtual void CollectRollouts() = 0;

        virtual std::array<torch::Tensor, 2> Predict(const torch::Tensor& observation, bool deterministic = false, const torch::Tensor& state = torch::Tensor(), const torch::Tensor& episode_start = torch::Tensor()) = 0;

        virtual void Train() = 0;

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

        virtual void RegisterLogItems() {
            log_items_["num_updates"] = torch::Tensor();
        }

        virtual bool Proceed() {
            return time_steps_ < max_time_steps_;
        }
        
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

        virtual void Monitor() {
            if (monitor_interval_ <= 0 || num_iters_ % monitor_interval_ != 0)
                return;
            PrintLog(); 
            if (!output_path_.empty())
                SaveLog(output_path_ + "_log.txt");
        }

        virtual void Checkpoint() {
            if (checkpoint_interval_ <= 0 || num_iters_ % checkpoint_interval_ != 0)
                return; 
            if (!output_path_.empty())
                Save(output_path_ + "_" + rlop::GetDatetime() + "_" + std::to_string(time_steps_) + ".pth");
        }

        virtual void Update() {
            ++num_iters_;
        }

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

        virtual void Load(const std::string& path, const std::unordered_set<std::string>& names = {"all"}) {
            torch::serialize::InputArchive archive;
            archive.load_from(path);
            LoadArchive(&archive, names); 
        }
        
        virtual void Save(const std::string& path, const std::unordered_set<std::string>& names = {"all"}) {
            torch::serialize::OutputArchive archive;
            SaveArchive(&archive, names);
            archive.save_to(path);
        }

        virtual void LoadArchive(torch::serialize::InputArchive* archive, const std::unordered_set<std::string>& names) {}

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