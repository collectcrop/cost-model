#include <cstddef>
#include <cstdint>
namespace osm_rmi {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 3088;
const uint64_t BUILD_TIME_NS = 12378355082;
const char NAME[] = "osm_rmi";
uint64_t lookup(uint64_t key, size_t* err);
}
