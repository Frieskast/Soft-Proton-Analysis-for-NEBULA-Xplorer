#include "mirror_single_paraboloid.hh"
#include "G4Polycone.hh"
#include "G4RotationMatrix.hh"
#include "G4PVPlacement.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include <vector>
#include <cmath>
#include <sstream>

void BuildNestedParaboloids(
    const MirrorParams& params, G4LogicalVolume* logicWorld,
    G4Material* gold, G4Material* epoxy, G4Material* aluminium)
{
    // --- Constants ---
    const double goldThickness = 0.0002 * CLHEP::mm;
    const double epoxyThickness = 0.012 * CLHEP::mm;
    const double aluminiumThickness = 0.152 * CLHEP::mm;
    const double shellGap = 1.0 * CLHEP::mm;
    const double rMax = params.rMax;
    const double rMin = params.rMin;
    const double shell_thickness = goldThickness + epoxyThickness + aluminiumThickness;

    // --- Geometry extents ---
    const double x_start = 0.75 * CLHEP::m;
    const double x_end   = 0.85 * CLHEP::m;
    const double x_mid   = 0.80 * CLHEP::m; // Middle of the shell
    const double FL = params.focalLength;
    const int N = 125;

    // --- Start with outermost shell ---
    double prev_y_tip = rMax; // at x_end
    double prev_y_end = -1;   // will be set after first shell
    int shellIdx = 0;

    while (true) {
        // 1. Solve for p so that prev_y_tip^2 = p(2*x_end + p)
        double p = -x_end + std::sqrt(x_end * x_end + prev_y_tip * prev_y_tip);
        double y_end = std::sqrt(p * (2 * x_start + p));

        // --- Prevent shells too close to rMin ---
        if (y_end < rMin + shellGap + shell_thickness) break;

        // Print shell parameters
        G4cout << "Paraboloid shell " << shellIdx
               << ": p = " << p/CLHEP::mm << " mm, "
               << "y_tip(x_end) = " << prev_y_tip/CLHEP::mm << " mm, "
               << "y_end(x_start) = " << y_end/CLHEP::mm << " mm"
               << G4endl;

        // Stop if the shell end would go below rMin
        if (y_end < rMin) break;

        // Build the three layers: gold (inner), epoxy (middle), aluminium (outer)
        double layer_inner[3] = {y_end, y_end + goldThickness, y_end + goldThickness + epoxyThickness};
        double layer_outer[3] = {y_end + goldThickness, y_end + goldThickness + epoxyThickness, y_end + goldThickness + epoxyThickness + aluminiumThickness};
        G4Material* layer_mat[3] = {gold, epoxy, aluminium};
        const char* layer_name[3] = {"Gold", "Epoxy", "Aluminium"};

        for (int layer = 0; layer < 3; ++layer) {
            std::vector<double> xList, rInnerList, rOuterList;
            for (int i = 0; i <= N; ++i) {
                double x = x_start + (x_end - x_start) * i / N;
                double y = std::sqrt(p * (2 * x + p));
                xList.push_back(x);
                rInnerList.push_back(y + (layer_inner[layer] - y_end));
                rOuterList.push_back(y + (layer_outer[layer] - y_end));
            }
            std::ostringstream solidName;
            solidName << "SingleParabolic_Primary_" << shellIdx << "_" << layer_name[layer];
            G4Polycone* shell = new G4Polycone(
                solidName.str().c_str(), 0, 2*M_PI, N+1,
                xList.data(), rInnerList.data(), rOuterList.data()
            );
            std::ostringstream lvName;
            lvName << "SingleParabolic_PrimaryLV_" << shellIdx << "_" << layer_name[layer];
            G4LogicalVolume* logicShell = new G4LogicalVolume(shell, layer_mat[layer], lvName.str().c_str());

            G4RotationMatrix* rotY90 = new G4RotationMatrix();
            rotY90->rotateY(-90.*CLHEP::deg);

            std::ostringstream pvName;
            pvName << "SingleParabolic_PrimaryPV_" << shellIdx << "_" << layer_name[layer];

            new G4PVPlacement(rotY90, G4ThreeVector(0, 0, 0), logicShell, pvName.str().c_str(), logicWorld, false, shellIdx*3+layer, true);

            // Visualization
            G4Colour col;
            if (layer == 0)      col = G4Colour(1.0, 0.84, 0.0); // Gold/yellow
            else if (layer == 1) col = G4Colour(1.0, 0.84, 0.0);   // Blue (epoxy)
            else                 col = G4Colour(0.7, 0.7, 0.7);   // Grey (aluminium)
            G4VisAttributes* shellVis = new G4VisAttributes(col);
            shellVis->SetForceSolid(true);
            logicShell->SetVisAttributes(shellVis);
        }

        // --- Find next shell's y_tip at x_end so that the previous shell's OUTER surface
        // and the next shell's INNER surface are separated by at least shellGap.
        // Use a binary search to find the largest next_y_tip (i.e. closest) that satisfies the constraint.
        double min_tip = rMin + shellGap;                // cannot go inside this
        double max_tip = prev_y_tip - 1e-12;             // cannot exceed previous tip
        double next_y_tip = min_tip;

        // previous shell outer surfaces (inner radii + full material thickness)
        double prev_outer_tip = prev_y_tip + shell_thickness;
        double prev_outer_end = (prev_y_end < 0) ? -1.0 : (prev_y_end + shell_thickness);

        if (max_tip > min_tip) {
            double lo = min_tip;
            double hi = max_tip;
            double best = min_tip;
            for (int it = 0; it < 60; ++it) { // 60 iterations -> sub-nm precision, plenty
                double mid = 0.5 * (lo + hi);
                double mid_p = -x_end + std::sqrt(x_end * x_end + mid * mid);
                double mid_y_end = std::sqrt(mid_p * (2 * x_start + mid_p));

                // separation between previous outer surface and next inner surface at tip and at end
                double sep_tip = prev_outer_tip - mid;
                double sep_end = (prev_y_end < 0) ? 1e9 : (prev_outer_end - mid_y_end);

                if (sep_tip >= shellGap && sep_end >= shellGap) {
                    // mid is acceptable -> try to move closer (increase mid)
                    best = mid;
                    lo = mid;
                } else {
                    // not acceptable -> move inward (decrease mid)
                    hi = mid;
                }
            }
            next_y_tip = best;
        }
        // ensure within bounds
        if (next_y_tip < min_tip) next_y_tip = min_tip;
        if (next_y_tip > prev_y_tip) next_y_tip = prev_y_tip;

        // compute parabola parameters for the selected next_y_tip
        double final_p = -x_end + std::sqrt(x_end * x_end + next_y_tip * next_y_tip);
        double final_y_end = std::sqrt(final_p * (2 * x_start + final_p));

        // verify parabola identity for diagnostics using final_p
        for (int i=0;i<=N;++i) {
          double x = x_start + (x_end-x_start)*i/N;
          double y = std::sqrt(final_p*(2*x+final_p));
          double resid = y*y - final_p*(2*x+final_p); // should be ~0
          if (std::fabs(resid) > 1e-9) {
            G4cout<<"PARABOLA RESIDUAL large: "<<resid<<G4endl;
            break;
          }
        }
        // print separations (mm) using final_y_end
        double sep_tip = (prev_outer_tip - next_y_tip)/CLHEP::mm;
        double sep_end = (prev_outer_end - final_y_end)/CLHEP::mm;
        G4cout<<"sep_tip(mm)="<<sep_tip<<" sep_end(mm)="<<sep_end<<G4endl;

        prev_y_end = y_end;
        prev_y_tip = next_y_tip;
        shellIdx++;
    }
}
