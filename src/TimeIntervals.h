#pragma once

#include <utility>
#include <vector>

#include <petscsystypes.h>

class TimeIntervals
{
public:
    TimeIntervals(const std::string& ints);

    using data_t = std::vector<std::pair<PetscReal, PetscReal>>;
    using iterator = data_t::iterator;

    iterator begin() { return intervals.begin(); }
    iterator end() { return intervals.end(); }
    data_t::value_type operator[](const size_t i) { return intervals[i]; }

    bool inside(const PetscReal t) const;

private:
    std::vector<std::pair<PetscReal, PetscReal>> intervals;
};
