#include <cstddef>
#include <cstdint>
namespace my_first_rmi {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 2416;
const uint64_t BUILD_TIME_NS = 469299964;
const char NAME[] = "my_first_rmi";
uint64_t lookup(uint64_t key, size_t* err);
}
