#ifndef MYSTEPPINGACTION_HH
#define MYSTEPPINGACTION_HH

#include "G4UserSteppingAction.hh"
#include "G4Accumulable.hh"
#include <map>

class MySteppingAction : public G4UserSteppingAction {
public:
    MySteppingAction();
    virtual ~MySteppingAction();

    virtual void UserSteppingAction(const G4Step* step);

    void ResetLastVolumeMap() const;  // Mark as const

private:
    mutable std::map<G4int, G4String> lastVolumeMap;  // Use mutable to allow modification in const methods
    std::map<G4int, bool> recordedTracks;            // Track whether a track ID has been recorded
};

#endif