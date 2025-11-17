#include <cstddef>
#include <cstdint>
namespace books_rmi {
bool load(char const* dataPath);
void cleanup();
const size_t RMI_SIZE = 3088;
const uint64_t BUILD_TIME_NS = 25918498824;
const char NAME[] = "books_rmi";
uint64_t lookup(uint64_t key, size_t* err);
}
