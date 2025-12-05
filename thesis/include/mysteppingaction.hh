#ifndef MYSTEPPINGACTION_HH
#define MYSTEPPINGACTION_HH

#include "G4UserSteppingAction.hh"
#include "G4ThreeVector.hh"
#include <map>
#include <vector>
#include <string>
#include <set> // <-- Add this

// Define a structure to hold information about a single reflection event.
struct ReflectionHit {
    G4ThreeVector position;
    std::string materialName;
    std::string volumeName;
};

class MySteppingAction : public G4UserSteppingAction {
public:
    MySteppingAction();
    virtual ~MySteppingAction();
    virtual void UserSteppingAction(const G4Step* step) override;
    
    // --- MODIFICATION: Use a single map for all mirror component hits ---
    std::map<G4int, std::vector<ReflectionHit>> trackMirrorHits;
    std::map<G4int, std::vector<G4ThreeVector>> trackTrajectories;
    std::map<G4int, std::set<G4String>> trackEnteredMirrors;

private:
    // No private members needed now
};

#endif