#pragma once
#include "env.h"
#include "rlop/common/base_algorithm.h"
#include "rlop/common/selectors.h"
 
namespace multi_armed_bandit {
	class MultiArmedBanditSolver : public rlop::BaseAlgorithm {
	public:
		MultiArmedBanditSolver(const std::string& name, Env* env, double alpha = 0) : 
			name_(name),
			env_(env),
			alpha_(alpha)
		{}

		virtual ~MultiArmedBanditSolver() = default;

		virtual std::optional<Int> Select() = 0;

		virtual void Reset() {
			q_values_ = std::vector<double>(env_->action_space().Size(), 0);
			num_visits_ = std::vector<Int>(env_->action_space().Size(), 0);
		}

		virtual void Reset(uint64_t seed) {
			Reset();
			rand_.Seed(seed);
		}

		virtual void Solve(Int max_num_steps) {
			num_steps_ = 0;	
			max_num_steps_ = max_num_steps;
			total_rewards_ = std::vector<double>(max_num_steps, 0);
			num_opts_ = std::vector<Int>(max_num_steps, 0);
			while(Proceed()) {
				if (!Step())
					break;
				Update();
			}
		}

		virtual bool Proceed() {
			return num_steps_ < max_num_steps_;
		}

		virtual void Update() {
			++num_steps_;
		}

		virtual void UpdateQValue(Int i, double reward) {
			if (alpha_ > 0) 
				q_values_[i] += (reward - q_values_[i]) * alpha_;
			else
				q_values_[i] += (reward - q_values_[i]) / (num_visits_[i] + 1);	
		}

		virtual bool Step() {
			auto action_i = Select();
			if (!action_i)
				return false;
			double reward = env_->Step(*action_i);
			++num_visits_[*action_i];
			UpdateQValue(*action_i, reward);
			total_rewards_[num_steps_] = env_->total_reward();
			num_opts_[num_steps_] = (env_->action_space().Get(*action_i) == env_->best_arm());
			return true;
		};

		const std::string& name() const {
			return name_;
		}

		Int num_steps() const {
			return num_steps_;
		}

		const std::vector<double>& total_rewards() const {
			return total_rewards_;
		}

		const std::vector<Int>& num_opts() const {
			return num_opts_;
		}
		
	protected:
		std::string name_;
		Env* env_ = nullptr;
		Int num_steps_ = 0;
		Int max_num_steps_ = 0;
		double alpha_;
		std::vector<double> q_values_;
		std::vector<Int> num_visits_;
		std::vector<double> total_rewards_;
		std::vector<Int> num_opts_;
		rlop::Random rand_;
	}; 

	class EpsilonGreedySolver : public MultiArmedBanditSolver {
	public:
		EpsilonGreedySolver(const std::string& name, Env* env, double epsilon = 0.1, double alpha = 0) : 
			MultiArmedBanditSolver(name, env, alpha),
			epsilon_(epsilon)
		{}

		virtual ~EpsilonGreedySolver() = default;

		virtual std::optional<Int> Select() override {
			if (rand_.Uniform(0.0, 1.0) >= epsilon_)
				return rlop::SelectBest(q_values_.begin(), q_values_.end());
			else
				return { rand_.Uniform(Int(0), env_->action_space().Size() - 1) };
		}	
	
	protected:
		double epsilon_;
	};

	class SoftmaxSolver : public MultiArmedBanditSolver {
	public:
		SoftmaxSolver(const std::string& name, Env* env, double temp = 1.0, double alpha = 0) : 
			MultiArmedBanditSolver(name, env, alpha),
			temp_(temp)
		{}

		virtual ~SoftmaxSolver() = default;

		virtual std::optional<Int> Select() override {
			auto probs = rlop::Softmax(q_values_.begin(), q_values_.end(), temp_);
			if (probs.empty())
				return std::nullopt;
			Int i = rand_.Discrete<Int>(probs.begin(), probs.end());
			return { i };
		}	

	protected:
		double temp_;
	};

	class UCB1Solver : public MultiArmedBanditSolver {
	public:
		UCB1Solver(const std::string& name, Env* env, double c = std::sqrt(2), double alpha = 0) : 
			MultiArmedBanditSolver(name, env, alpha),
			c_(c)
		{}

		virtual ~UCB1Solver() = default;

		virtual std::optional<Int> Select() override {
			std::vector<double> scores(env_->action_space().Size());
			for (Int i=0; i<scores.size(); ++i) {
				scores[i] = rlop::UCB1(q_values_[i], num_visits_[i], num_steps_, c_);
			}
			return rlop::SelectBest(scores.begin(), scores.end());
		}	

	protected:
		double c_;
	};

	class PersuitSolver : public MultiArmedBanditSolver {
	public:
		PersuitSolver(const std::string& name, Env* env, double beta = 0.01, double alpha = 0) :
			MultiArmedBanditSolver(name, env, alpha),
			beta_(beta)
		{}

		virtual void Reset() override {
			MultiArmedBanditSolver::Reset();	
			prefs_ = std::vector<double>(env_->action_space().Size(), 1.0 / env_->action_space().Size());
		}

		virtual std::optional<Int> Select() override {
			Int i = rand_.Discrete<Int>(prefs_.begin(), prefs_.end());
			return { i };
		}

		virtual void UpdatePrefs() {
			auto max_q = std::max_element(q_values_.begin(), q_values_.end());
			Int best_i = std::distance(q_values_.begin(), max_q);
			for (Int i=0; i<prefs_.size(); ++i) {
				if (i == best_i)
					prefs_[i] += beta_ * (1 - prefs_[i]);
				else 
					prefs_[i] += beta_ * (0 - prefs_[i]);
			}
		}

		virtual void Update() override {
			UpdatePrefs();
			++num_steps_;
		}

	protected:
		double beta_;
		std::vector<double> prefs_;
	};

	class PursuitEpsilonGreedySolver : public PersuitSolver {
	public:
		PursuitEpsilonGreedySolver(const std::string& name, Env* env, double epsilon = 0.1, double beta = 0.01, double alpha = 0) :
			PersuitSolver(name, env, beta, alpha),
			epsilon_(epsilon)
		{}

		virtual std::optional<Int> Select() override {
			if (rand_.Uniform(0.0, 1.0) >= epsilon_)
				return rlop::SelectBest(prefs_.begin(), prefs_.end());
			else
				return { rand_.Uniform(Int(0), env_->action_space().Size()-1) };
		}	

	protected:
		double epsilon_;
	};
}