#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <ctime>
#include "rlop/common/utils.h"
#include "rlop/common/random.h"

namespace snake {
    using rlop::Int;
    using rlop::kIntNull;
    using rlop::kIntFull;

    class Engine {
    public:
        static constexpr Int kUp = 0;
        static constexpr Int kDown = 1;
        static constexpr Int kLeft = 2;
        static constexpr Int kRight = 3;

        struct Tile {
            Int num_snakes = 0;
            bool has_food = false;
        };

        struct Snake {
            Snake(Int x, Int y, Int dir) : body({{x, y}}), dir(dir) {}

            std::vector<std::pair<Int, Int>> body;
            Int dir = kIntNull;
            Int len = 1;
            Int num_foods = 0;
            bool alive = true;
        };

        Engine(Int num_snakes = 1)  :
            num_snakes_(num_snakes),
            num_alives_(num_snakes),
            grid_(grid_height_, std::vector<Tile>(grid_width_))
        {}

        virtual ~Engine() = default;

        virtual void Reset() {
            num_steps_ = 0;
            num_alives_ = num_snakes_;
            grid_ = std::vector<std::vector<Tile>>(grid_height_, std::vector<Tile>(grid_width_));
            foods_.clear();
            SetSnakes();
            SetFoods();
        }

        virtual void SetSnakes() {
            snakes_.clear();
            std::vector<Int> tiles(grid_width_ * grid_height_);
            std::iota(tiles.begin(), tiles.end(), 0);
            rand_.PartialShuffle(tiles.begin(), tiles.end(), num_snakes_);
            for (Int i=0; i<num_snakes_; ++i) {
                Int y = tiles[i] / grid_width_;
                Int x = tiles[i] % grid_width_;
                Int dir = rand_.Uniform(0, 3);
                snakes_.emplace_back(x, y, dir);
                ++grid_[y][x].num_snakes;
            }
        }

        virtual void SetFoods() {
            if (foods_.size() >= min_num_foods_)
                return;
            std::vector<std::pair<Int, Int>> pos;
            for (Int x = 0; x < grid_width_; ++x) {
                for (Int y=0; y < grid_height_; ++y) {
                    if (!grid_[y][x].has_food && grid_[y][x].num_snakes == 0) 
                        pos.push_back({x, y});
                }
            }
            Int num_foods = min_num_foods_ - foods_.size();
            rand_.PartialShuffle(pos.begin(), pos.end(), num_foods);
            for (Int i=0; i<num_foods; ++i) {
                grid_[pos[i].second][pos[i].first].has_food = true;
                foods_.push_back(pos[i]);
            }
        }

        virtual std::pair<Int, Int> GetNextPos(const std::pair<Int, Int>& pos, Int dir) const {
            auto new_pos = pos;
            if (dir == kUp)
                new_pos.second = (pos.second == 0? grid_height_: pos.second) - 1;
            else if (dir == kDown)
                new_pos.second = (pos.second + 1) % grid_height_;
            else if (dir == kLeft)
                new_pos.first =  (pos.first == 0? grid_width_: pos.first) - 1;
            else if (dir == kRight)
                new_pos.first = (pos.first + 1) % grid_width_;
            return new_pos;
            // if (dir == kUp)
            //     new_pos.second = pos.second - 1;
            // else if (dir == kDown)
            //     new_pos.second = pos.second + 1;
            // else if (dir == kLeft)
            //     new_pos.first = pos.first - 1;
            // else if (dir == kRight)
            //     new_pos.first = pos.first + 1;
            // return new_pos;
        }

        virtual bool OutOfBoundary(const std::pair<Int,Int>& pos) const {
            return pos.first < 0 || pos.first >= grid_width_ || pos.second < 0 || pos.second >= grid_height_;
        }

        virtual bool CheckCollision(const std::pair<Int,Int>& pos) const {
            return OutOfBoundary(pos) || grid_[pos.second][pos.first].num_snakes > 0;
        }

        virtual bool Lookahead(Int snake_i, Int dir) const {
            if (dir  == GetReverseDir(snakes_[snake_i].dir))
                return false;
            auto head = GetNextPos(snakes_[snake_i].body.front(), dir);
            if (CheckCollision(head))
                return false;
            return true;
        }

        virtual const std::pair<Int, Int>& GetHead(Int snake_i) const {
            return snakes_[snake_i].body.front();
        }

        virtual Int GetMinFoodDistance(const std::pair<Int, Int>& head) const {
            Int min_dist = std::numeric_limits<Int>::max();
            for (const auto& pos : foods_) {
                Int dist_x = std::min(std::abs(pos.first - head.first), grid_width_ - std::abs(pos.first - head.first));
                Int dist_y = std::min(std::abs(pos.second - head.second), grid_height_ - std::abs(pos.second - head.second));
                Int dist = dist_x + dist_y;
                if (dist < min_dist)
                    min_dist = dist;
            }
            return min_dist;
        }

        virtual Int GetWinner() const {
            Int best_i = kIntNull;
            Int max_len = 0;
            for (Int i=0; i<snakes_.size(); ++i) {
                if (snakes_[i].alive && snakes_[i].len > max_len) {
                    max_len = snakes_[i].len;
                    best_i = i;
                }
            }
            return best_i;
        }

        virtual void Update() {
            if (num_steps_ >= max_num_steps_)
                return;
            std::vector<bool> to_remove(num_snakes_, false);
            for (Int i=0; i<snakes_.size(); ++i) {
                if (!snakes_[i].alive)
                    continue;
                auto head = GetNextPos(snakes_[i].body.front(), snakes_[i].dir);
                if (OutOfBoundary(head)) {
                    snakes_[i].alive = false;
                    to_remove[i] = true;
                    --num_alives_;
                }
                else {
                    snakes_[i].body.insert(snakes_[i].body.begin(), head);
                    ++grid_[head.second][head.first].num_snakes;
                }
            }
            for (Int i=0; i<snakes_.size(); ++i) {
                if (!snakes_[i].alive)
                    continue;
                auto head = snakes_[i].body.front();
                if (grid_[head.second][head.first].num_snakes > 1) {
                    snakes_[i].alive = false;
                    to_remove[i] = true;
                    --num_alives_;
                }
                else if (grid_[head.second][head.first].has_food) {
                    ++snakes_[i].len;
                    ++snakes_[i].num_foods;
                }
            }
            if (num_steps_ > 0 && num_steps_ % hunger_rate_ == 0) {
                for (Int i=0; i<snakes_.size(); ++i) {
                    if (!snakes_[i].alive)
                        continue;
                    if (snakes_[i].len == 1) {
                        snakes_[i].alive = false;
                        to_remove[i] = true;
                        --num_alives_;
                    }
                    else {
                        --grid_[snakes_[i].body.back().second][snakes_[i].body.back().first].num_snakes;
                        snakes_[i].body.pop_back();
                        --snakes_[i].len;
                    }
                }
            }
            for (Int i=0; i<snakes_.size(); ++i) {
                if (!snakes_[i].alive && to_remove[i]) {
                    for (auto& p : snakes_[i].body) {
                        --grid_[p.second][p.first].num_snakes;
                    }
                }
                else if (snakes_[i].len < snakes_[i].body.size()) {
                    --grid_[snakes_[i].body.back().second][snakes_[i].body.back().first].num_snakes;
                    snakes_[i].body.pop_back();
                }
                else {
                    std::pair<Int, Int> head = snakes_[i].body.front();
                    for (Int i=0; i<foods_.size(); ++i) {
                        if (foods_[i].first == head.first && foods_[i].second == head.second) {
                            foods_.erase(foods_.begin() + i);
                            grid_[head.second][head.first].has_food = false;
                            break;
                        }
                    }
                }
            }
            SetFoods();
            ++num_steps_;
        }

        virtual Int GetReverseDir(Int dir) const {
            switch(dir) {
            case kUp:
                return kDown;
            case kDown:
                return kUp;
            case kLeft:
                return kRight;
            case kRight:
                return kLeft;
            }
            return kIntNull;
        }

        virtual void SetDir(Int i, Int dir) {
            if (dir == GetReverseDir(snakes_[i].dir))
                return;
            snakes_[i].dir = dir;
        }

        virtual bool IsStart() const {
            return num_steps_ == 0;
        }

        virtual bool IsEnd() const {
            return num_alives_ <= 0 || num_steps_ >= max_num_steps_;
        }

        Int grid_width() const {
            return grid_width_;
        }

        Int grid_height() const {
            return grid_height_;
        }

        Int grid_size() const {
            return grid_size_;
        }

        Int min_num_foods() const {
            return min_num_foods_;
        }
        
        Int max_num_steps() const {
            return max_num_steps_;
        }

        Int num_steps() const {
            return num_steps_;
        }

        Int hunger_rate() const {
            return hunger_rate_;
        }
        
        Int num_alives() const {
            return num_alives_;
        }

        const std::vector<std::vector<Tile>>& grid() const {
            return grid_;
        } 

        const std::vector<Snake>& snakes() const {
            return snakes_;
        }

        const std::vector<std::pair<Int, Int>>& foods() const {
            return foods_;
        }

        const rlop::Random& rand() const {
            return rand_;
        }

        void set_seed(uint64_t seed) {
            rand_.Seed(seed);
        }

    private:
        Int grid_width_ = 11;
        Int grid_height_ = 7;
        Int grid_size_ = grid_width_ * grid_height_;
        Int min_num_foods_ = 2;
        Int max_num_steps_ = 200;
        Int hunger_rate_ = 40;
        Int num_snakes_ = 0;
        Int num_alives_ = 0;
        Int num_steps_ = 0;
        std::vector<std::vector<Tile>> grid_;
        std::vector<Snake> snakes_;
        std::vector<std::pair<Int, Int>> foods_;
        rlop::Random rand_;
    };

    class Graphics {
    public:
        Graphics() = default;
        
        Graphics(const Engine& engine, Int fps = 20, Int tile_size = 50) : 
            tile_size_(tile_size),
            fps_(fps)
        {
            engines_.push_back(&engine);
        }

        Graphics(const std::vector<const Engine*>& engines, Int fps = 20, Int tile_size = 50) : 
            engines_(engines),
            tile_size_(tile_size),
            fps_(fps)
        {}

        Graphics(std::vector<const Engine*>&& engines, Int fps = 20, Int tile_size = 50) : 
            engines_(std::move(engines)),
            tile_size_(tile_size),
            fps_(fps)
        {}

        virtual ~Graphics() = default;

        virtual void Reset() {
            layout_shape_ = GetLayoutShape(engines_.size());
            sub_grid_shape_ = { engines_.front()->grid_width(), engines_.front()->grid_height()};
            sub_window_shape_ = {sub_grid_shape_.first * tile_size_ + 2 * gap_, sub_grid_shape_.second * tile_size_ + 2 * gap_};
            const unsigned int window_width = layout_shape_.first * (sub_window_shape_.first + gap_) - gap_;
            const unsigned int window_height = layout_shape_.second * (sub_window_shape_.second + gap_) - gap_;
            std::cout << window_width << " " << window_height << std::endl;
            window_.create(sf::VideoMode(window_width, window_height), "Snake");
            window_.setFramerateLimit(fps_);
            Int num_snakes = engines_.front()->snakes().size();
            food_rect_ = sf::RectangleShape(sf::Vector2f(tile_size_, tile_size_));
            food_rect_.setFillColor(sf::Color::White);
            snake_rect_ = sf::RectangleShape(sf::Vector2f(tile_size_, tile_size_));
            snake_head_ = sf::CircleShape(tile_size_ / 2.0);
        }

        virtual std::pair<Int, Int> GetLayoutShape(Int num_engines) {
            Int num_rows = std::sqrt(num_engines);
            Int num_cols = std::ceil((double)num_engines / num_rows);
            return { num_cols, num_rows };
        }

        virtual bool IsOpen() const {
            return window_.isOpen();
        }

        virtual void HandleEvents() {
            while (window_.pollEvent(event_)) {
                if (event_.type == sf::Event::Closed)
                    window_.close();
            }
        }

        virtual void Run() {
            while (IsOpen()) {
                HandleEvents();
                Render();
            }
        }

        virtual void Render() {
            window_.clear();
            Int engine_i = 0;
            for (Int i = 0; i < layout_shape_.first; ++i) {
                for (Int j = 0; j < layout_shape_.second; ++j) {
                    Int sub_x = i * (sub_window_shape_.first + gap_);
                    Int sub_y = j * (sub_window_shape_.second + gap_);

                    for (Int xi = 0; xi <= sub_grid_shape_.first; ++xi) {
                        Int line_x = sub_x + gap_ + xi * tile_size_;
                        sf::Vertex line[] = {
                            sf::Vertex(sf::Vector2f(line_x, sub_y + gap_)),
                            sf::Vertex(sf::Vector2f(line_x, sub_y + gap_ + sub_grid_shape_.second * tile_size_))
                        };
                        window_.draw(line, 2, sf::Lines);
                    }

                    for (Int yi = 0; yi <= sub_grid_shape_.second; ++yi) {
                        Int line_y = sub_y + gap_ + yi * tile_size_;
                        sf::Vertex line[] = {
                            sf::Vertex(sf::Vector2f(sub_x + gap_, line_y)),
                            sf::Vertex(sf::Vector2f(sub_x + gap_ + sub_grid_shape_.first * tile_size_, line_y))
                        };
                        window_.draw(line, 2, sf::Lines);
                    }
                    if (engine_i < engines_.size()) {
                        for (auto& p : engines_[engine_i]->foods()) {
                            food_rect_.setPosition(sub_x + gap_ + p.first * tile_size_, sub_y + gap_ + p.second * tile_size_);
                            window_.draw(food_rect_);
                        }
                        for (Int snake_i=0; snake_i<engines_[engine_i]->snakes().size(); ++snake_i) {
                            if (!engines_[engine_i]->snakes()[snake_i].alive)
                                continue;
                            auto& body = engines_[engine_i]->snakes()[snake_i].body.front();
                            if (snake_i == 0) 
                                snake_head_.setFillColor(sf::Color::Green); 
                            else if (snake_i == 1)
                                snake_head_.setFillColor(sf::Color::Blue); 
                            else if (snake_i == 2)
                                snake_head_.setFillColor(sf::Color::Red); 
                            else
                                snake_head_.setFillColor(sf::Color::Cyan);  
                            snake_head_.setPosition(sub_x + gap_ + body.first * tile_size_, sub_y + gap_ + body.second * tile_size_);
                            window_.draw(snake_head_);
                            for (Int body_i=1; body_i<engines_[engine_i]->snakes()[snake_i].body.size(); ++body_i) {
                                auto& body = engines_[engine_i]->snakes()[snake_i].body[body_i];
                                if (snake_i == 0) 
                                    snake_rect_.setFillColor(sf::Color(0, std::max(255 - 2*body_i, Int(30)), 0, std::max(255 - 2*body_i, Int(30)))); 
                                else if (snake_i == 1)
                                    snake_rect_.setFillColor(sf::Color(0, 0, std::max(255 - 2*body_i, Int(30)), std::max(255 - 2*body_i, Int(30)))); 
                                else if (snake_i == 2)
                                    snake_rect_.setFillColor(sf::Color(std::max(255 - 2*body_i, Int(30)), 0, 0, std::max(255 - 2*body_i, Int(30)))); 
                                else
                                    snake_rect_.setFillColor(sf::Color(0, std::max(255 - 2*body_i, Int(30)), std::max(255 - 2*body_i, Int(30)), std::max(255 - 2*body_i, Int(30)))); 
                                snake_rect_.setPosition(sub_x + gap_ + body.first * tile_size_, sub_y + gap_ + body.second * tile_size_);
                                window_.draw(snake_rect_);
                            }
                        }
                    }
                    ++engine_i;
                }
            }
            window_.display();
        }

        Int tile_size() const {
            return tile_size_;
        }

        Int fps() const {
            return fps_;
        }

    protected:
        Int tile_size_ = 0;
        Int fps_ = 0;
        Int gap_ = 8;
        std::pair<Int, Int> layout_shape_;
        std::pair<Int, Int> sub_grid_shape_;
        std::pair<Int, Int> sub_window_shape_;
        std::vector<const Engine*> engines_;
        sf::RenderWindow window_;
        sf::Event event_;
        sf::RectangleShape food_rect_;
        sf::RectangleShape snake_rect_;
        sf::CircleShape snake_head_;
    };

    class Game {
    public:
        Game(Int num_snakes = 1, Int fps = 20) : engine_(num_snakes), graphics_(engine_, fps) {}

        void Reset() {
            engine_.Reset();
            input_ = kIntNull;
        }
        
        bool IsOpen() const {
            return graphics_.IsOpen();
        }

        void Run() {
            while (graphics_.IsOpen()) {
                graphics_.HandleEvents();
                GetInput();
                if (clock_.getElapsedTime().asSeconds() > 0.3) {
                    if (input_ != kIntNull)
                        engine_.SetDir(0, input_);
                    engine_.Update();
                    graphics_.Render();
                    clock_.restart();
                }
            }
        }

        void GetInput() {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) 
                input_ = Engine::kUp;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
                input_ = Engine::kDown;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
                input_ = Engine::kLeft;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
                input_ = Engine::kRight;
        }

    protected:
        Int input_ = kIntNull;
        Engine engine_;
        Graphics graphics_;
        sf::Clock clock_;
    };
}


