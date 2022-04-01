#include "TimeIntervals.h"

#include <limits>
#include <sstream>

#include "TimeIntervals.h"

TimeIntervals::TimeIntervals(const std::string& ints)
{
    std::istringstream iss(ints);
    iss.ignore(1, '[');
    while (iss.peek() != ']')
    {
        PetscReal start;
        PetscReal end;
        iss >> start;
        iss.ignore(1, '-');
        iss >> end;
        if (iss.peek() == ',')
        {
            iss.ignore(1, ',');
        }
        intervals.emplace_back(std::make_pair(start, end));
    }
}

bool TimeIntervals::inside(const PetscReal t) const
{
    for (const auto& interval : intervals)
    {
        if (t >= interval.first and t <= interval.second)
        {
            return true;
        }
    }
    return false;
}
