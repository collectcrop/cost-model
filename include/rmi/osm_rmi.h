#include <cstddef>
#include <cstdint>
namespace osm_rmi {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 12582944;
const uint64_t BUILD_TIME_NS = 14332049145;
const char NAME[] = "osm_rmi";
uint64_t lookup(uint64_t key, size_t* err);
}
