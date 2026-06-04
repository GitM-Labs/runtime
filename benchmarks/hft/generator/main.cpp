#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>
#include <parquet/properties.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>

static constexpr double  HAWKES_MU    = 100.0;
static constexpr double  HAWKES_ALPHA = 0.6;
static constexpr double  HAWKES_BETA  = 0.8;
static int64_t CHUNK = 5000000;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: hft_gen <n_events> <seed> <out_path>\n";
        return 1;
    }
    int64_t     n_events = std::stoll(argv[1]);
    uint64_t    seed     = std::stoull(argv[2]);
    std::string out      = argv[3];
    if (argc >= 5) CHUNK = std::stoll(argv[4]);

    std::cout << "Generating " << n_events << " events, seed=" << seed << "\n";

    auto schema = arrow::schema({
        arrow::field("ts_ns",     arrow::int64()),
        arrow::field("symbol_id", arrow::int32()),
        arrow::field("side",      arrow::int8()),
        arrow::field("price",     arrow::int64()),
        arrow::field("size",      arrow::int32()),
        arrow::field("type",      arrow::int8()),
    });

    auto props = parquet::WriterProperties::Builder()
        .compression(parquet::Compression::ZSTD)->compression_level(1)
        ->build();
    auto arrow_props = parquet::ArrowWriterProperties::Builder().build();

    auto outfile_res = arrow::io::FileOutputStream::Open(out);
    if (!outfile_res.ok()) { std::cerr << outfile_res.status().ToString() << "\n"; return 1; }
    auto outfile = outfile_res.ValueOrDie();

    auto writer_res = parquet::arrow::FileWriter::Open(
        *schema, arrow::default_memory_pool(), outfile, props, arrow_props);
    if (!writer_res.ok()) { std::cerr << writer_res.status().ToString() << "\n"; return 1; }
    auto writer = std::move(writer_res).ValueOrDie();

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double>  uni(0.0, 1.0);
    std::uniform_int_distribution<int32_t>  sym_d(0, 99);
    std::uniform_int_distribution<int32_t>  sz_d(1, 1000);
    std::uniform_int_distribution<int8_t>   side_d(0, 1);
    std::uniform_int_distribution<int8_t>   type_d(0, 2);

    double  t         = 0.0;
    double  lambda    = HAWKES_MU;
    int64_t mid_price = 100000;
    int64_t written   = 0;

    while (written < n_events) {
        int64_t count = std::min(CHUNK, n_events - written);

        arrow::Int64Builder ts_b, price_b;
        arrow::Int32Builder sym_b, size_b;
        arrow::Int8Builder  side_b, type_b;

        for (int64_t i = 0; i < count; i++) {
            double dt = -std::log(uni(rng)) / lambda;
            t      += dt;
            lambda  = HAWKES_MU + (lambda - HAWKES_MU) * std::exp(-HAWKES_BETA * dt) + HAWKES_ALPHA;
            int8_t type = type_d(rng);
            if (type == 2) mid_price += (side_d(rng) == 0 ? -1 : 1);

            (void)ts_b.Append((int64_t)(t * 1e9));
            (void)sym_b.Append(sym_d(rng));
            (void)side_b.Append(side_d(rng));
            (void)price_b.Append(mid_price + (side_d(rng) == 0 ? -1 : 1));
            (void)size_b.Append(sz_d(rng));
            (void)type_b.Append(type);
        }

        std::shared_ptr<arrow::Array> ts_a, sym_a, side_a, price_a, size_a, type_a;
        (void)ts_b.Finish(&ts_a);   (void)sym_b.Finish(&sym_a);
        (void)side_b.Finish(&side_a); (void)price_b.Finish(&price_a);
        (void)size_b.Finish(&size_a); (void)type_b.Finish(&type_a);

        auto batch = arrow::RecordBatch::Make(schema, count,
            {ts_a, sym_a, side_a, price_a, size_a, type_a});
        auto table = arrow::Table::FromRecordBatches({batch}).ValueOrDie();
        (void)writer->WriteTable(*table, count);

        written += count;
        std::cout << "Progress: " << written << "/" << n_events << "\r" << std::flush;
    }

    (void)writer->Close();
    std::cout << "\nDone. " << written << " events written.\n";
    return 0;
}
