#include "generator.hh"
#include "G4Event.hh" 
#include "Randomize.hh" 
#include "G4SystemOfUnits.hh" 
#include "G4ProductionCutsTable.hh" 
#include "G4RegionStore.hh" 
#include "G4ProductionCuts.hh" 
#include "runaction.hh"
#include "G4RunManager.hh"
#include "generatorMessenger.hh"
#include "G4ParticleTable.hh"
#include "G4Gamma.hh"
#include "G4Proton.hh"
#include "G4GeneralParticleSource.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Tubs.hh"
#include "G4Box.hh"
#include "construction.hh" 
#include <optional>

MyPrimaryGenerator::MyPrimaryGenerator()
{
    fParticleGun = new G4ParticleGun(1); // Initialize particle gun with 1 particle per event

    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();

    // Generate a proton
    G4String particleName = "proton";
    G4ParticleDefinition* particle = particleTable->FindParticle(particleName);

    // Set the particle energy
    fParticleGun->SetParticleEnergy(1000 * keV); 

    // Set the particle definition
    fParticleGun->SetParticleDefinition(particle);

    fMessenger = new GeneratorMessenger(this);

    fGPS = new G4GeneralParticleSource();
}

MyPrimaryGenerator::~MyPrimaryGenerator()
{
    delete fMessenger;
    delete fParticleGun;
    delete fGPS;
}

void MyPrimaryGenerator::GeneratePrimaries(G4Event* anEvent)
{
    // small adjustable gap after the filter (set to 0 if you want no gap)
    const double apertureGap = 1.0 * mm;

    // Compute apertureX using detector knowledge. Only query physical volumes if the detector
    // says filters are present (avoids GetVolume warnings when no filter is placed).
    G4double apertureX = 0.0;
    const G4VUserDetectorConstruction* baseDetector =
        G4RunManager::GetRunManager()->GetUserDetectorConstruction();
    auto* myDet = dynamic_cast<const MyDetectorConstruction*>(baseDetector);
    if (myDet && myDet->GetThermalFilterOn()) {
        // detector reports filters present -> safe to find their PVs
        auto getMaxFaceX = [](const G4String& pvName)->std::optional<G4double> {
            G4VPhysicalVolume* pv = G4PhysicalVolumeStore::GetInstance()->GetVolume(pvName);
            if (!pv) return std::nullopt;
            G4ThreeVector t = pv->GetObjectTranslation();
            G4VSolid* solid = pv->GetLogicalVolume()->GetSolid();
            if (auto* tubs = dynamic_cast<G4Tubs*>(solid)) {
                return t.x() + tubs->GetZHalfLength();
            }
            if (auto* box = dynamic_cast<G4Box*>(solid)) {
                return t.x() + box->GetXHalfLength();
            }
            return t.x();
        };
        std::optional<G4double> polyFace = getMaxFaceX("FilterPolyPV");
        std::optional<G4double> alFace   = getMaxFaceX("FilterAlPV");
        if (polyFace || alFace) {
            G4double maxFace = -1e30;
            if (polyFace) maxFace = std::max(maxFace, *polyFace);
            if (alFace)   maxFace = std::max(maxFace, *alFace);
            apertureX = maxFace + apertureGap;
        } else {
            // Filters supposedly on but PVs not found: fall back to stop plate or default
            G4VPhysicalVolume* plate = G4PhysicalVolumeStore::GetInstance()->GetVolume("StopPlatePV");
            if (plate) {
                G4double plateX = plate->GetObjectTranslation().x();
                apertureX = (std::abs(plateX - 0.85*m) < 0.02*m) ? 0.85*m + apertureGap : 0.90*m + apertureGap;
            } else {
                apertureX = 0.85*m + apertureGap;
            }
        }
       
    } else {
        // No filter -> use fallback positions (no GetVolume calls -> no warnings)
        G4VPhysicalVolume* plate = G4PhysicalVolumeStore::GetInstance()->GetVolume("StopPlatePV");
        if (plate) {
            G4double plateX = plate->GetObjectTranslation().x();
            apertureX = (std::abs(plateX - 0.85*m) < 0.02*m) ? 0.85*m + apertureGap : 0.90*m + apertureGap;
        } else {
            apertureX = 0.85*m + apertureGap;
        }
    }
    // G4cout << "Aperture X coordinate set to: " << apertureX / mm << " mm" << G4endl;
    // G4cout << "aperture x coordinate" << apertureX << G4endl;
    // aperture radii (tunable / could be exposed via messenger)
    const double rout = 0.047 * m;
    const double rin  = 0.0 * m;

    // compute cone half-angle for limiting directions (optional)
    const double theta_max = std::atan2(rout, apertureX);

    // --- Sample position uniformly on disk/annulus at x = apertureX ---
    double u = G4UniformRand();
    double r = std::sqrt(u * (rout * rout - rin * rin) + rin * rin);
    double phi = 2.0 * CLHEP::pi * G4UniformRand();
    double y = r * std::cos(phi);
    double z = r * std::sin(phi);
    G4ThreeVector pos(apertureX, y, z);
    fParticleGun->SetParticlePosition(pos);

    // --- Sample direction for isotropic flux through a plane: PDF ∝ cosθ·sinθ ---
    double mu0 = std::cos(theta_max);             // cos(theta_max)
    double u2 = G4UniformRand();                  // uniform [0,1)
    // sample mu = cos(theta) with PDF that yields p(θ) ∝ cosθ·sinθ
    double mu = std::sqrt(std::max(0.0, 1.0 - u2 * (1.0 - mu0*mu0)));
    double phi_dir = 2.0 * CLHEP::pi * G4UniformRand();

    // components for axis = -x (particles travel toward focus at origin)
    double cos_theta = mu;
    double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    double dx = -cos_theta;
    double dy = sin_theta * std::cos(phi_dir);
    double dz = sin_theta * std::sin(phi_dir);
    G4ThreeVector dir(dx, dy, dz);
    dir = dir.unit();

    fParticleGun->SetParticleMomentumDirection(dir);
    fParticleGun->GeneratePrimaryVertex(anEvent);
}

void MyPrimaryGenerator::SetParticleType(const G4String& name) {
    G4ParticleDefinition* particle = G4ParticleTable::GetParticleTable()->FindParticle(name);
    if (particle) {
        fParticleGun->SetParticleDefinition(particle);
    } else {
        G4Exception("MyPrimaryGenerator::SetParticleType", "InvalidParticle", JustWarning,
                    ("Unknown particle: " + name).c_str());
    }
}

