#ifndef MIRROR_WOLTER1_HH
#define MIRROR_WOLTER1_HH

#include "construction.hh"

// Forward declarations
class G4LogicalVolume;
class G4Material;

void BuildNestedWolterI(const MirrorParams&, G4LogicalVolume*, G4Material*, G4Material*, G4Material*);

#endif