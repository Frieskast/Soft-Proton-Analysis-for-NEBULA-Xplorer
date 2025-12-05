#ifndef TRACKINFO_HH
#define TRACKINFO_HH
#include "G4VUserTrackInformation.hh"

class MyTrackInfo : public G4VUserTrackInformation {
public:
    MyTrackInfo() : G4VUserTrackInformation(), touchedGold(false) {}
    bool touchedGold;
};

#endif