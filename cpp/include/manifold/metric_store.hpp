#pragma once

/// @file metric_store.hpp
/// @brief Persistent storage and caching layer for MetricTensor instances.
///
/// MetricStore manages the lifecycle of MetricTensor objects across charts,
/// providing:
///   - Thread-safe in-memory cache (reader-writer lock via shared_mutex)
///   - File-based persistence (binary .bin files per chart)
///   - Batch metric evaluation across multiple points
///   - Double-checked locking for efficient lazy loading
///
/// Storage layout:
///   {db_path}/
///     metric_0.bin    ← chart 0's metric tensor
///     metric_1.bin    ← chart 1's metric tensor
///     ...
///
/// Each .bin file contains: [chart_id:u32][dim:u32][serialized MetricTensor payload]

#include "metric_tensor.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace manifold {

// ═══════════════════════════════════════════════════════════════════════════════
//  MetricStore  –  Thread-safe persistence for MetricTensors
// ═══════════════════════════════════════════════════════════════════════════════

/// Manages persistent storage and caching of MetricTensor instances.
///
/// Thread safety:
///   Uses std::shared_mutex for reader-writer locking. Multiple concurrent
///   reads (get_metric, batch_evaluate) are allowed; writes (commit, create_metric)
///   acquire exclusive locks. Double-checked locking pattern for lazy loading.
class MetricStore {
public:
    /// Construct a MetricStore backed by a directory on disk.
    /// Creates the directory if it does not exist.
    /// @param db_path  Directory path for metric storage files
    explicit MetricStore(const std::string& db_path)
        : db_path_(db_path)
    {
        std::filesystem::create_directories(db_path_);
    }

    /// Retrieve the metric tensor for a chart.
    ///
    /// Search order:
    ///   1. In-memory cache (shared lock)
    ///   2. Disk file at {db_path}/metric_{chart_id}.bin (exclusive lock)
    ///
    /// @param chart_id  Chart to look up
    /// @return  Shared pointer to MetricTensor, or nullptr if not found
    [[nodiscard]] std::shared_ptr<MetricTensor> get_metric(uint32_t chart_id) {
        // Fast path: check cache under shared (reader) lock
        {
            std::shared_lock lock(mutex_);
            auto it = cache_.find(chart_id);
            if (it != cache_.end()) return it->second;
        }

        // Slow path: load from disk under exclusive (writer) lock
        std::unique_lock wlock(mutex_);
        // Double-check after acquiring write lock (another thread may have loaded)
        auto it = cache_.find(chart_id);
        if (it != cache_.end()) return it->second;

        // Attempt load from disk – try common dim values
        for (uint32_t d = 1; d <= 1024; d *= 2) {
            auto m = load_metric(chart_id, d);
            if (m) {
                cache_[chart_id] = m;
                return m;
            }
        }
        return nullptr;
    }

    /// Create a new identity metric tensor for a chart and register it.
    /// If a metric already exists for this chart, it is replaced.
    ///
    /// @param chart_id  Chart identifier
    /// @param dim       Intrinsic dimension d
    /// @return  Shared pointer to the newly created MetricTensor
    std::shared_ptr<MetricTensor> create_metric(uint32_t chart_id, uint32_t dim) {
        auto m = std::make_shared<MetricTensor>(chart_id, dim);
        m->set_identity();

        std::unique_lock lock(mutex_);
        cache_[chart_id] = m;
        return m;
    }

    /// Persist a metric tensor to storage and update the cache.
    ///
    /// Serialises the metric to disk and performs a deep copy into the
    /// in-memory cache (ensuring cache and disk are consistent).
    ///
    /// @param chart_id  Chart identifier
    /// @param metric    MetricTensor to persist
    void commit(uint32_t chart_id, const MetricTensor& metric) {
        // Write to disk first (outside lock for concurrency)
        save_metric(metric);

        // Update cache under exclusive lock
        std::unique_lock lock(mutex_);
        auto copy = std::make_shared<MetricTensor>(chart_id, metric.dim());
        auto data  = metric.serialize();
        copy->deserialize(data);
        cache_[chart_id] = copy;
    }

    /// Batch evaluate the metric at multiple points on the same chart.
    ///
    /// Thread-safe: acquires a shared lock for the duration of evaluation.
    ///
    /// @param chart_id  Chart identifier
    /// @param points    Vector of local coordinate vectors
    /// @return  Vector of d × d metric matrices (one per point)
    [[nodiscard]] std::vector<Matrix> batch_evaluate(
            uint32_t chart_id,
            const std::vector<Vector>& points) const
    {
        std::shared_lock lock(mutex_);
        auto it = cache_.find(chart_id);
        if (it == cache_.end()) return {};

        const auto& m = it->second;
        std::vector<Matrix> results;
        results.reserve(points.size());
        for (const auto& p : points) {
            results.push_back(m->evaluate(p));
        }
        return results;
    }

    // ── Storage Management ────────────────────────────────────────────────────

    /// Number of charts currently in the cache.
    [[nodiscard]] size_t num_charts() const {
        std::shared_lock lock(mutex_);
        return cache_.size();
    }

    /// Check whether a metric for the given chart exists (in cache).
    [[nodiscard]] bool has_chart(uint32_t chart_id) const {
        std::shared_lock lock(mutex_);
        return cache_.find(chart_id) != cache_.end();
    }

    /// Flush all cached metrics to disk.
    void flush() {
        std::unique_lock lock(mutex_);
        for (auto& [id, m] : cache_) {
            save_metric(*m);
        }
    }

private:
    std::string db_path_;   ///< Base directory for metric .bin files
    mutable std::shared_mutex mutex_;   ///< Reader-writer lock
    std::unordered_map<uint32_t, std::shared_ptr<MetricTensor>> cache_;

    // ── File I/O ──────────────────────────────────────────────────────────────

    /// Write a MetricTensor to a binary file.
    /// File format: [chart_id:u32][dim:u32][serialized_payload:bytes]
    void save_metric(const MetricTensor& metric) const {
        auto data  = metric.serialize();
        std::string path = db_path_ + "/metric_" + std::to_string(metric.chart_id()) + ".bin";

        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) {
            throw SerializationError("Cannot open metric file for writing: " + path);
        }

        // Header
        uint32_t cid = metric.chart_id();
        uint32_t dim = metric.dim();
        ofs.write(reinterpret_cast<const char*>(&cid), sizeof(cid));
        ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

        // Payload
        ofs.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size()));
    }

    /// Load a MetricTensor from a binary file.
    /// @param chart_id  Expected chart ID (must match file header)
    /// @param dim       Expected dimension (must match file header)
    /// @return  Shared pointer to loaded MetricTensor, or nullptr on failure
    std::shared_ptr<MetricTensor> load_metric(uint32_t chart_id, uint32_t dim) {
        std::string path = db_path_ + "/metric_" + std::to_string(chart_id) + ".bin";
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return nullptr;

        // Read and validate header
        uint32_t file_cid = 0, file_dim = 0;
        ifs.read(reinterpret_cast<char*>(&file_cid), sizeof(file_cid));
        ifs.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
        if (file_cid != chart_id || file_dim != dim) return nullptr;

        // Read payload
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(ifs)),
                                   std::istreambuf_iterator<char>());

        try {
            auto m = std::make_shared<MetricTensor>(chart_id, dim);
            m->deserialize(data);
            return m;
        } catch (const DBException&) {
            return nullptr;
        }
    }
};

} // namespace manifold
