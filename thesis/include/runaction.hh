#ifndef RUNACTION_HH
#define RUNACTION_HH

#include "G4UserRunAction.hh"
#include "G4AnalysisManager.hh"
#include "G4Accumulable.hh"
#include "G4AccumulableManager.hh"
#include <chrono>

class G4Run;

class MyRunAction : public G4UserRunAction {
public:
    MyRunAction();
    virtual ~MyRunAction();

    virtual void BeginOfRunAction(const G4Run* run);
    virtual void EndOfRunAction(const G4Run* run);
    // Reflection counters API (thread-safe via G4Accumulable)
    void IncrementReflectedCount();
    void IncrementReflectedHitCount();
    G4int GetReflectedCount() const;
    G4int GetReflectedHitCount() const;

    void AddExitCount(G4int n) { fExitCount += n; }
    G4int GetIncidentCount() const { return fIncidentCount; }
    void IncrementExitCount() { fExitCount += 1; }
    G4int GetExitCount() const { return fExitCount.GetValue(); }
    G4String GenerateFileName();

    // Setters and getters for run parameters
    void SetIncidentEnergy(G4double e) { fIncidentEnergy = e; }
    G4double GetIncidentEnergy() const { return fIncidentEnergy; }
    void SetIncidentAngle(double angleDeg) { fIncidentAngleDeg = angleDeg; }
    double GetIncidentAngle() const { return fIncidentAngleDeg; }
    void SetEnergy(double energyKeV) { fEnergyKeV = energyKeV; }
    double GetEnergy() const { return fEnergyKeV; }
    void SetMirrorConfig(const G4String& config) { fMirrorConfig = config; }
    G4String GetMirrorConfig() const { return fMirrorConfig; }
    void SetFilterOn(bool on) { fFilterOn = on; }
    bool GetFilterOn() const { return fFilterOn; }

    // energy accumulators API
    void AddEdepDet(G4double e) { fEdepDet += e; }
    void AddEdepFocal(G4double e) { fEdepFocal += e; }
    G4double GetEdepDet() const { return fEdepDet.GetValue(); }
    G4double GetEdepFocal() const { return fEdepFocal.GetValue(); }

private:
    G4Accumulable<G4int> fExitCount;
    // reflection accumulables
    G4Accumulable<G4int> fReflectedCount;
    G4Accumulable<G4int> fReflectedHitCount;

    // energy accumulables (keV)
    G4Accumulable<G4double> fEdepDet;
    G4Accumulable<G4double> fEdepFocal;
    G4int fIncidentCount = 0;
    G4double fIncidentEnergy = 0;
    double fIncidentAngleDeg = 0;
    double fEnergyKeV = 0;
    std::string fPendingFileName;
    G4String fMirrorConfig;
    bool fFilterOn = true; // default ON
    std::chrono::time_point<std::chrono::high_resolution_clock> fStartTime;
    G4int fNEvents = 0;
    // id for the run-level summary ntuple (created in ctor)
    G4int fRunSummaryNtupleId = -1;
};

#endif
