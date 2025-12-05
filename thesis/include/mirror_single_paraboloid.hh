#ifndef MIRROR_SINGLE_PARABOLOID_HH
#define MIRROR_SINGLE_PARABOLOID_HH

#include "construction.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"

void BuildNestedParaboloids(
    const MirrorParams& params,
    G4LogicalVolume* logicWorld,
    G4Material* gold,
    G4Material* epoxy,
    G4Material* aluminium);

#endif
