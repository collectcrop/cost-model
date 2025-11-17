#include <cstddef>
#include <cstdint>
namespace wiki_rmi {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 3088;
const uint64_t BUILD_TIME_NS = 29367093673;
const char NAME[] = "wiki_rmi";
uint64_t lookup(uint64_t key, size_t* err);
}
