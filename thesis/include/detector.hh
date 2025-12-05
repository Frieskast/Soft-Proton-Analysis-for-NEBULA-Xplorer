#ifndef DETECTOR_HH
#define DETECTOR_HH

#include "G4VSensitiveDetector.hh"
#include "G4Step.hh"
#include "G4ThreeVector.hh"
#include "G4Types.hh"
#include <map>
#include <unordered_map>
#include <set> // Include the set header

class MySensitiveDetector : public G4VSensitiveDetector {
public:
    MySensitiveDetector(G4String name);
    virtual ~MySensitiveDetector();

    virtual void Initialize(G4HCofThisEvent*) override;
    virtual G4bool ProcessHits(G4Step *aStep, G4TouchableHistory *ROhist) override;
    virtual void EndOfEvent(G4HCofThisEvent*) override;

    // make these thread_local so worker threads don't race on plain statics
    static thread_local G4int fDetectorHitCount;
    static thread_local G4double total_edep;
    static thread_local G4double total_edep_det;
    static thread_local G4double total_edep_focal;

    // Print TID in krad at end of run
    static void PrintTIDinKrad();

    // Access last computed run-level doses (krad) written by PrintTIDinKrad
    // Return value < 0.0 indicates "not set".
    static double GetLastDoseKradDet();
    static double GetLastDoseKradFocal();

    // (previous helper removed — use the RunAction / PrintTIDinKrad codepaths for dose)

    // --- per-track maps (thread_local) - public so EventAction can access/clear ---
    static thread_local std::map<G4int, G4double> siDetKineticEnergies;
    static thread_local std::map<G4int, G4double> entryKineticEnergies;
    static thread_local std::map<G4int, G4double> lastKineticEnergies;

    // per-event (thread_local) energy sums so EndOfEventAction can access exact per-event edep
    static thread_local std::unordered_map<G4int, G4double> perEventEdepDet;   // stores MeV
    static thread_local std::unordered_map<G4int, G4double> perEventEdepFocal; // stores MeV

    // Counter for detailed proton tracks
    static thread_local std::set<G4int> s_detailedProtonTrackIDs;
    static const G4int kMaxDetailedTracks = 10;
private:
    static double s_lastDoseKradDet;
    static double s_lastDoseKradFocal;
};

#endif
