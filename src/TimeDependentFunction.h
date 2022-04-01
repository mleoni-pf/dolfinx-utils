#pragma once

#include <petscsystypes.h>

#include <dolfinx/fem/Function.h>

#include "DirichletBCDefinitions.h"

class TimeDependentFunction
{
public:
    void setTime(const PetscReal t) { time = t; }
    virtual DirichletBCFunction asInterpolableFunction() const = 0;
    virtual bool isActive() const { return true; }

protected:
    PetscReal time;
};
