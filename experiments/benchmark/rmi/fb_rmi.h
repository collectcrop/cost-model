#include <cstddef>
#include <cstdint>
namespace fb_rmi {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 3088;
const uint64_t BUILD_TIME_NS = 29735199164;
const char NAME[] = "fb_rmi";
uint64_t lookup(uint64_t key, size_t* err);
}
