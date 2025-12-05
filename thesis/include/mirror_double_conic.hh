#ifndef MIRROR_DOUBLE_CONIC_HH
#define MIRROR_DOUBLE_CONIC_HH

#include "construction.hh"

// Forward declarations to reduce header dependencies
class G4LogicalVolume;
class G4Material;

void BuildDoubleConic(const MirrorParams&, G4LogicalVolume*, G4Material*, G4Material*, G4Material*);

#endif