#include "mirror_double_conic.hh"
#include "G4Polycone.hh"
#include "G4RotationMatrix.hh"
#include "G4PVPlacement.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4Material.hh"      // <-- Add this
#include "G4LogicalVolume.hh"  // <-- Add this
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip> // Add this at the top for std::setprecision
#include <fstream> // For file I/O
#include <sys/stat.h> // For checking file existence

// Helper function to check if a file exists
inline bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

void BuildDoubleConic(
    const MirrorParams& params,
    G4LogicalVolume* logicWorld,
    G4Material* gold,
    G4Material* epoxy,
    G4Material* aluminium)
{
    const double FL = params.focalLength;
    const double Lprimary = 0.1 * CLHEP::m;
    const double Lsecondary = 0.1 * CLHEP::m;
    const double rMax = params.rMax;
    const double Rdumbell = params.rMin;
    const double goldThickness = 0.0002 * CLHEP::mm;
    const double epoxyThickness = 0.012 * CLHEP::mm;
    const double aluminiumThickness = 0.152 * CLHEP::mm;
    const double shellGap = 1.0 * CLHEP::mm;
    const int N = 125;

    double layer_inner[3] = {0, goldThickness, goldThickness + epoxyThickness};
    double layer_outer[3] = {goldThickness, goldThickness + epoxyThickness, goldThickness + epoxyThickness + aluminiumThickness};
    G4Material* layer_mat[3] = {gold, epoxy, aluminium};
    const char* layer_name[3] = {"Gold", "Epoxy", "Aluminium"};

    int shellIdx = 0;
    const double shell_thickness = goldThickness + epoxyThickness + aluminiumThickness;

    double prev_Ysmin = 1e9; // large value for first shell
    const double Pi_step = 0.0001 * CLHEP::mm; // 0.1 micron step

    // --- CSV file setup ---
    const std::string geo_filename = "mirror_geometry.csv";
    bool write_file = !file_exists(geo_filename);
    std::ofstream geo_file;
    if (write_file) {
        geo_file.open(geo_filename);
        geo_file << "name,z_points,r_points\n"; // Write header
    }

    // Find Pi_new for first shell so that Ypmax = rMax
    double Pi_first = 0.0;
    {
        // Ypmax = Pi_new + ((2*FL + Lprimary) - (2*FL + 0.5*Lprimary)) * tan(theta)
        // where theta = atan(Pi_new / FL) / 4.0
        // Solve for Pi_new numerically
        double Pi_lo = Rdumbell;
        double Pi_hi = rMax;
        for (int iter = 0; iter < 100; ++iter) {
            double Pi_try = 0.5 * (Pi_lo + Pi_hi);
            double theta_try = std::atan(Pi_try / FL) / 4.0;
            double tan_theta = std::tan(theta_try);
            double Ypmax_try = Pi_try + ((2 * FL + Lprimary) - (2 * FL + 0.5 * Lprimary)) * tan_theta;
            if (std::abs(Ypmax_try - rMax) < 1e-6 * CLHEP::mm) {
                Pi_first = Pi_try;
                break;
            }
            if (Ypmax_try > rMax)
                Pi_hi = Pi_try;
            else
                Pi_lo = Pi_try;
            Pi_first = Pi_try;
        }
    }
    double Pi_new = Pi_first; // Only declare here!

    while (true) {
        double theta_i, tan_theta, tan_3theta;
        double Ypmin, Ypmax, Ysmin, Ysmax;

        // Find next shell with at least shellGap spacing between Ysmins
        while (true) {
            theta_i = std::atan(Pi_new / FL) / 4.0;
            tan_theta = std::tan(theta_i);
            tan_3theta = std::tan(3 * theta_i);

            Ypmin = Pi_new + ((2 * FL) - (2 * FL + 0.5 * Lprimary)) * tan_theta;
            Ypmax = Pi_new + ((2 * FL + Lprimary) - (2 * FL + 0.5 * Lprimary)) * tan_theta;
            Ysmin = Ypmin + (2 * FL - (2 * FL + Lprimary)) * tan_3theta;
            Ysmax = Ypmin; // at x = 2*FL

            // For the first shell, always accept
            if (shellIdx == 0) break;

            // For subsequent shells, require Ysmin spacing
            if ((prev_Ysmin - Ysmin) >= shellGap  + shell_thickness) break;

            Pi_new -= Pi_step;
            if (Pi_new < Rdumbell) goto done; // stop if below min radius
        }

        // Stop if shell would go below Rdumbell + shellGap
        if (Ysmin - shell_thickness < Rdumbell + shellGap) break;

        // --- Build layers for primary cone ---
        for (int layer = 0; layer < 3; ++layer) {
            std::vector<double> xP, rInnerP, rOuterP;
            for (int j = 0; j <= N; ++j) {
                double x = (2 * FL + Lprimary) + (0 - Lprimary) * j / N;
                double y = Pi_new + (x - (2 * FL + 0.5 * Lprimary)) * tan_theta;
                xP.push_back(x);
                rInnerP.push_back(y + layer_inner[layer]);
                rOuterP.push_back(y + layer_outer[layer]);
            }

            // --- Write geometry to CSV if needed ---
            if (write_file && layer == 0) { // Only for the inner gold layer
                std::ostringstream lvName;
                lvName << "DoubleConic_PrimaryLV_" << shellIdx << "_" << layer_name[layer];
                geo_file << lvName.str() << ",";
                // The G4Polycone is defined along its own x-axis, then rotated and placed.
                // World Z = Local X - FL
                for (size_t i = 0; i < xP.size(); ++i) {
                    geo_file << (xP[i] - FL) / CLHEP::mm << (i == xP.size() - 1 ? "" : " ");
                }
                geo_file << ",";
                // World X/Y = Local R
                for (size_t i = 0; i < rInnerP.size(); ++i) {
                    geo_file << rInnerP[i] / CLHEP::mm << (i == rInnerP.size() - 1 ? "" : " ");
                }
                geo_file << "\n";
            }

            std::ostringstream solidName;
            solidName << "DoubleConic_Primary_" << shellIdx << "_" << layer_name[layer];
            G4Polycone* shell = new G4Polycone(
                solidName.str().c_str(), 0, 2*M_PI, N+1,
                xP.data(), rInnerP.data(), rOuterP.data()
            );
            std::ostringstream lvName;
            lvName << "DoubleConic_PrimaryLV_" << shellIdx << "_" << layer_name[layer];
            G4LogicalVolume* logicShell = new G4LogicalVolume(shell, layer_mat[layer], lvName.str().c_str());

            G4RotationMatrix* rotY90 = new G4RotationMatrix();
            rotY90->rotateY(-90.*CLHEP::deg);

            new G4PVPlacement(rotY90, G4ThreeVector(-FL, 0, 0), logicShell, solidName.str().c_str(), logicWorld, false, shellIdx*6+layer, true);

            G4Colour col;
            if (layer == 0)      col = G4Colour(1.0, 0.84, 0.0);
            else if (layer == 1) col = G4Colour(1.0, 0.84, 0.0);
            else                 col = G4Colour(0.7, 0.7, 0.7);
            G4VisAttributes* shellVis = new G4VisAttributes(col);
            shellVis->SetForceSolid(true);
            logicShell->SetVisAttributes(shellVis);
        }

        // --- Build layers for secondary cone ---
        for (int layer = 0; layer < 3; ++layer) {
            std::vector<double> xS, rInnerS, rOuterS;
            for (int j = 0; j <= N; ++j) {
                double x = (2 * FL) + (0 - Lsecondary) * j / N;
                double y = Ypmin + (x - (2 * FL)) * tan_3theta;
                xS.push_back(x);
                rInnerS.push_back(y + layer_inner[layer]);
                rOuterS.push_back(y + layer_outer[layer]);
            }

            // --- NEW: Write geometry to CSV if needed ---
            if (write_file && layer == 0) { // Only for the inner gold layer
                std::ostringstream lvName;
                lvName << "DoubleConic_SecondaryLV_" << shellIdx << "_" << layer_name[layer];
                geo_file << lvName.str() << ",";
                // World Z = Local X - FL
                for (size_t i = 0; i < xS.size(); ++i) {
                    geo_file << (xS[i] - FL) / CLHEP::mm << (i == xS.size() - 1 ? "" : " ");
                }
                geo_file << ",";
                // World X/Y = Local R
                for (size_t i = 0; i < rInnerS.size(); ++i) {
                    geo_file << rInnerS[i] / CLHEP::mm << (i == rInnerS.size() - 1 ? "" : " ");
                }
                geo_file << "\n";
            }

            std::ostringstream solidName;
            solidName << "DoubleConic_Secondary_" << shellIdx << "_" << layer_name[layer];
            G4Polycone* shell = new G4Polycone(
                solidName.str().c_str(), 0, 2*M_PI, N+1,
                xS.data(), rInnerS.data(), rOuterS.data()
            );
            std::ostringstream lvName;
            lvName << "DoubleConic_SecondaryLV_" << shellIdx << "_" << layer_name[layer];
            G4LogicalVolume* logicShell = new G4LogicalVolume(shell, layer_mat[layer], lvName.str().c_str());

            G4RotationMatrix* rotY90 = new G4RotationMatrix();
            rotY90->rotateY(-90.*CLHEP::deg);

            new G4PVPlacement(rotY90, G4ThreeVector(-FL, 0, 0), logicShell, solidName.str().c_str(), logicWorld, false, shellIdx*6+3+layer, true);

            G4Colour col;
            if (layer == 0)      col = G4Colour(1.0, 0.84, 0.0);
            else if (layer == 1) col = G4Colour(1.0, 0.84, 0.0);
            else                 col = G4Colour(0.7, 0.7, 0.7);
            G4VisAttributes* shellVis = new G4VisAttributes(col);
            shellVis->SetForceSolid(true);
            logicShell->SetVisAttributes(shellVis);
        }

        // Print info for this shell
        G4cout << "DoubleConic shell " << shellIdx
               << ": Pi=" << Pi_new/CLHEP::mm << " mm, "
               << "Ypmin=" << Ypmin/CLHEP::mm << " mm, Ypmax=" << Ypmax/CLHEP::mm << " mm, "
               << "Ysmin=" << Ysmin/CLHEP::mm << " mm"
               << G4endl;

        // Prepare for next shell (move inward)
        prev_Ysmin = Ysmin;
        Pi_new -= shell_thickness; // move Pi_new inward by shell thickness for next shell
        shellIdx++;
    }
done:
    if (write_file) {
        geo_file.close();
    }
    G4cout << "BuildDoubleConic CALLED" << G4endl;
}
