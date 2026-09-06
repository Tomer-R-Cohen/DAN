#include "platform.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main(int argc, char* argv[])
{
    std::string error;
    assert(dan::platform::initialize(error));
    const std::string name = argc == 2 ? argv[1]
        : "DAN.Provider.test." + std::to_string(dan::platform::process_id());
    if (argc == 2) return dan::platform::acquire_single_instance(name) ? 1 : 0;
    assert(dan::platform::acquire_single_instance(name));
    dan::platform::Process duplicate;
    assert(duplicate.start({argv[0], name}, error));
    assert(duplicate.wait() == 0);
    dan::platform::cleanup();
}
