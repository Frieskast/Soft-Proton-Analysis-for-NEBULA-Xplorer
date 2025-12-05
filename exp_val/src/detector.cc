#include "detector.hh"
#include <fstream> 

// This script follows the proton after reflection and prints the data to the console and saves it to a hits_output.txt IS IT REDUNCANT?

MySensitiveDetector::MySensitiveDetector(G4String name) : G4VSensitiveDetector(name)
{
    // Clear the file at the start of the simulation
    std::ofstream outFile("hits_output.txt", std::ios::trunc);
    outFile << "Hits Output File\n";
    outFile.close();
}

MySensitiveDetector::~MySensitiveDetector()
{}

G4bool MySensitiveDetector::ProcessHits(G4Step *aStep, G4TouchableHistory *ROhist)
{
    G4Track *track = aStep->GetTrack();

    // Stop the track after it hits the detector
    track->SetTrackStatus(fStopAndKill);

    // Get the particle type
    G4String particleName = track->GetDefinition()->GetParticleName();

    // Get the kinetic energy of the particle
    G4double kineticEnergy = track->GetKineticEnergy();

    // Get the pre-step point position
    G4StepPoint *preStepPoint = aStep->GetPreStepPoint();
    G4ThreeVector posProton = preStepPoint->GetPosition();

    // Get the detector's copy number
    const G4VTouchable *touchable = preStepPoint->GetTouchable();
    G4int copyNo = touchable->GetCopyNumber();

    // Get the detector's position
    G4VPhysicalVolume *physVol = touchable->GetVolume();
    G4ThreeVector posDetector = physVol->GetTranslation();

    // Print the information to the console
    G4cout << "Particle: " << particleName << G4endl;
    G4cout << "Kinetic Energy: " << kineticEnergy / CLHEP::keV << " keV" << G4endl;
    G4cout << "Copy number: " << copyNo << G4endl;
    G4cout << "Detector position: " << posDetector << G4endl;

    // Write the information to a file
    std::ofstream outFile("hits_output.txt", std::ios::app); // Open file in append mode
    outFile << "Particle: " << particleName
            << ", Kinetic Energy: " << kineticEnergy / CLHEP::keV << " keV"
            << ", Position: (" << posProton.x() / CLHEP::mm << ", " << posProton.y() / CLHEP::mm << ", " << posProton.z() / CLHEP::mm << ") mm"
            << ", Copy number: " << copyNo
            << ", Detector position: (" << posDetector.x() / CLHEP::mm << ", " << posDetector.y() / CLHEP::mm << ", " << posDetector.z() / CLHEP::mm << ") mm"
            << std::endl;
    outFile.close(); // Close the file

    return true;
}
