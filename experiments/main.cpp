#include <CLI/CLI.hpp>
#include <map>
#include "FALCON/utils/include.hpp"
#include "./run/config.hpp"
#include "./run/thread_bench.cpp"
#include "./run/epsilon_bench.cpp"
#include "./run/batch_bench.cpp"
#include "./run/worker_bench.cpp"
#include "./run/range_eps_bench.cpp"
#include "./run/range_single_bench.cpp"
#include "./run/join_bench.cpp"
int main(int argc, char** argv) {

    CLI::App app{"FALCON Benchmark Suite"};

    app.require_subcommand(1);

    BaseConfig base;
    falcon::CachePolicy policy;

    std::map<std::string, falcon::CachePolicy> policy_map = {
        {"none", falcon::CachePolicy::NONE},
        {"fifo", falcon::CachePolicy::FIFO},
        {"lru",  falcon::CachePolicy::LRU},
        {"lfu",  falcon::CachePolicy::LFU}
    };
    auto map = CLI::CheckedTransformer(policy_map, CLI::ignore_case);


    // =========================
    // Point Benchmark
    // =========================
    auto point_cmd = app.add_subcommand("point", "Point query benchmark");

    point_cmd->add_option("--dataset", base.dataset, "Dataset basename")->required();
    point_cmd->add_option("--keys", base.num_keys, "Number of keys")->required();
    point_cmd->add_option("--repeats", base.repeats);
    point_cmd->add_option("--memory", base.memory_mb);
    point_cmd->add_option("--baseline", base.baseline);

    // =========================
    // Epsilon Benchmark
    // =========================
    EpsilonConfig eps_cfg;
    auto eps_cmd = app.add_subcommand("epsilon", "Vary epsilon experiment");

    eps_cmd->add_option("--dataset", eps_cfg.dataset)->required();
    eps_cmd->add_option("--keys", eps_cfg.num_keys)->required();
    eps_cmd->add_option("--epsilon", eps_cfg.epsilons, "List of epsilon values")->required();
    eps_cmd->add_option("--memory", eps_cfg.memory_mb)->required();
    eps_cmd->add_option("--policy", eps_cfg.policy)->transform(map);
    // eps_cmd->add_option("--repeats", eps_cfg.repeats);
    
    // =========================
    // Threads Benchmark
    // =========================
    ThreadConfig thread_cfg;
    auto thread_cmd = app.add_subcommand("threads", "Vary thread count experiment");

    thread_cmd->add_option("--dataset", thread_cfg.dataset)->required();
    thread_cmd->add_option("--keys", thread_cfg.num_keys)->required();
    thread_cmd->add_option("--threads", thread_cfg.thread_counts, "List of thread counts")->required();
    thread_cmd->add_option("--policy", thread_cfg.policy)->transform(map);
    thread_cmd->add_option("--memory", thread_cfg.memory_mb);
    // thread_cmd->add_option("--repeats", thread_cfg.repeats);

    // =========================
    // Batch Benchmark
    // =========================
    BatchConfig batch_cfg;
    auto batch_cmd = app.add_subcommand("batch", "Vary batch size experiment");

    batch_cmd->add_option("--dataset", batch_cfg.dataset)->required();
    batch_cmd->add_option("--keys", batch_cfg.num_keys)->required();
    batch_cmd->add_option("--batch", batch_cfg.batch_sizes, "List of batch sizes")->required();
    batch_cmd->add_option("--memory", batch_cfg.memory_mb);
    batch_cmd->add_option("--policy", batch_cfg.policy)->transform(map);
    // batch_cmd->add_option("--repeats", batch_cfg.repeats);

    // =========================
    // Worker Benchmark
    // =========================
    WorkerConfig worker_cfg;
    auto worker_cmd = app.add_subcommand("worker", "Vary worker size experiment");

    worker_cmd->add_option("--dataset", worker_cfg.dataset)->required();
    worker_cmd->add_option("--keys", worker_cfg.num_keys)->required();
    worker_cmd->add_option("--workers", worker_cfg.worker_counts, "List of worker counts")->required();
    worker_cmd->add_option("--producers", worker_cfg.producer_counts, "List of producer counts")->required();
    worker_cmd->add_option("--memory", worker_cfg.memory_mb);
    worker_cmd->add_option("--policy", worker_cfg.policy)->transform(map);

    // =========================
    // Range Eps Benchmark
    // =========================
    RangeEpsConfig range_eps_cfg;
    auto range_eps_cmd = app.add_subcommand("rangeEps", "Vary epsilon range experiment");

    range_eps_cmd->add_option("--dataset", range_eps_cfg.dataset)->required();
    range_eps_cmd->add_option("--keys", range_eps_cfg.num_keys)->required();
    range_eps_cmd->add_option("--epsilon", range_eps_cfg.epsilons, "List of epsilon values")->required();
    range_eps_cmd->add_option("--memory", range_eps_cfg.memory_mb)->required();
    range_eps_cmd->add_option("--policy", range_eps_cfg.policy)->transform(map);

    // =========================
    // Range Single Benchmark
    // =========================
    RangeSigConfig range_sig_cfg;
    auto range_sig_cmd = app.add_subcommand("rangeSig", "Use range query to find single point (query must be ordered)");

    range_sig_cmd->add_option("--dataset", range_sig_cfg.dataset)->required();
    range_sig_cmd->add_option("--keys", range_sig_cfg.num_keys)->required();
    range_sig_cmd->add_option("--query", range_sig_cfg.query_file, "Query file name")->required();
    range_sig_cmd->add_option("--memory", range_sig_cfg.memory_mb)->required();
    range_sig_cmd->add_option("--policy", range_sig_cfg.policy)->transform(map);


    // =========================
    // Hybrid Join Benchmark
    // =========================
    JoinConfig join_cfg;
    auto join_cmd = app.add_subcommand("join", "Hybrid join experiment");

    join_cmd->add_option("--dataset", join_cfg.dataset)->required();
    join_cmd->add_option("--keys", join_cfg.num_keys)->required();
    join_cmd->add_option("--query", join_cfg.query_file, "Query file name")->required();
    join_cmd->add_option("--memory", join_cfg.memory_mb)->required();
    join_cmd->add_option("--policy", join_cfg.policy)->transform(map);

    CLI11_PARSE(app, argc, argv);

    // =========================
    // Dispatch
    // =========================

    if (*point_cmd) {
        // run_point(base);
    }
    else if (*eps_cmd) {
        run_epsilon(eps_cfg);
    }
    else if (*thread_cmd) {
        run_threads(thread_cfg);
    }
    else if (*batch_cmd) {
        run_batch(batch_cfg);
    }
    else if (*worker_cmd) {
        run_worker(worker_cfg);
    }
    else if (*range_eps_cmd) {
        run_range_eps(range_eps_cfg);
    }
    else if (*range_sig_cmd) {
        run_range_sig(range_sig_cfg);
    }
    else if (*join_cmd) {
        run_join(join_cfg);
    }
}
