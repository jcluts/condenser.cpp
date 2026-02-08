#ifndef __RNG_H__
#define __RNG_H__

#include <vector>

class RNG {
public:
    virtual ~RNG()                               = default;
    virtual void manual_seed(uint64_t seed)      = 0;
    virtual std::vector<float> randn(uint32_t n) = 0;
};

#endif  // __RNG_H__