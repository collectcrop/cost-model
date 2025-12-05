#include <cstddef>
#include <cstdint>
namespace fb_rmi {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 12582928;
const uint64_t BUILD_TIME_NS = 9525022422;
const char NAME[] = "fb_rmi";
uint64_t lookup(uint64_t key, size_t* err);
}
