#include "mirror_wolter1.hh"
#include "G4Polycone.hh"
#include "G4RotationMatrix.hh"
#include "G4PVPlacement.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include <vector>
#include <cmath>
#include <sstream>
#include "G4Tubs.hh"

void BuildNestedWolterI(
    const MirrorParams& params,
    G4LogicalVolume* logicWorld,
    G4Material* gold,
    G4Material* epoxy,
    G4Material* aluminium)
{
    // --- Constants ---
    const double FL = params.focalLength; // Focal length
    const double Lparabolic = 0.1 * CLHEP::m;
    const double Lhyperbolic = 0.1 * CLHEP::m;
    const double rMax = params.rMax;
    const double Rdumbell = params.rMin; 
    const double goldThickness = 0.0002 * CLHEP::mm;
    const double epoxyThickness = 0.012 * CLHEP::mm;
    const double aluminiumThickness = 0.152 * CLHEP::mm;
    const double shellGap = 1.0 * CLHEP::mm;
    const int N = 125;

    // --- Layer definitions ---
    double layer_inner[3] = {0, goldThickness, goldThickness + epoxyThickness};
    double layer_outer[3] = {goldThickness, goldThickness + epoxyThickness, goldThickness + epoxyThickness + aluminiumThickness};
    G4Material* layer_mat[3] = {gold, epoxy, aluminium};
    const char* layer_name[3] = {"Gold", "Epoxy", "Aluminium"};

    // --- Precompute Abbe sphere intersection points (OUTSIDE IN) ---
    std::vector<std::pair<double, double>> abbe_points;
    for (double y = rMax - goldThickness - epoxyThickness - aluminiumThickness; y > Rdumbell + shellGap; y -= 0.005 * CLHEP::mm) {
        double x = FL + std::sqrt(FL*FL - y*y);
        abbe_points.emplace_back(x, y);
    }

    // --- Shell construction ---
    int shellIdx = 0;
    double prev_Ypmax = rMax + shellGap; // Start just above rMax
    double prev_Yhmin = 1e9; // Large value for first shell
    const double shell_thickness = goldThickness + epoxyThickness + aluminiumThickness;

    for (size_t i = 0; i < abbe_points.size(); ++i) {
        double x_int = abbe_points[i].first;
        double y_int = abbe_points[i].second;

        // Wolter-I geometry as before
        double theta_max = std::asin(y_int / FL) / 4;
        double p = y_int * std::tan(theta_max);
        double c = FL / 2.0;
        double a = FL * ((2 * std::cos(2 * theta_max) - 1) / 2.0);
        double b = std::sqrt(c * c - a * a);

        // Paraboloid: x from x_int to x_int + Lparabolic
        double Xpmin = x_int;
        double Xpmax = Xpmin + Lparabolic;
        double Ypmax = std::sqrt(p * (2 * Xpmax + p));

        // Hyperboloid: x from x_int - Lhyperbolic to x_int
        double Xhmax = x_int;
        double Xhmin = Xhmax - Lhyperbolic;
        double Yhmin = b * std::sqrt(((Xhmin - c)*(Xhmin - c)) / (a*a) - 1.0);

        // 1. Use the definition of 2a as the difference in distances from a point
        //    on the hyperbola to the two foci F1=(0,0) and F2=(-FL,0).
        //    We use the intersection point (x_int, y_int).
        double dist_to_F1 = std::sqrt(x_int * x_int + y_int * y_int);
        double dist_to_F2 = std::sqrt((x_int + FL) * (x_int + FL) + y_int * y_int);
        double a_stable = (dist_to_F2 - dist_to_F1) / 2.0;

        // 2. Now calculate b^2 using the stable 'a'. c is still FL/2.0.
        double b2_stable = c * c - a_stable * a_stable;

        // 3. Calculate Rc at the tightest point of our segment (Xhmin, Yhmin)
        //    using the stable a and b^2.
        double x_rel_to_center = Xhmin - c;
        double y_rel_to_center = Yhmin;

        double a2_stable = a_stable * a_stable;
        double a4_stable = a2_stable * a2_stable;
        double b4_stable = b2_stable * b2_stable;

        double numerator = std::pow(a4_stable * y_rel_to_center * y_rel_to_center + b4_stable * x_rel_to_center * x_rel_to_center, 1.5);
        double denominator = a4_stable * b4_stable;
        double min_Rc_hyperboloid = (denominator > 0) ? (numerator / denominator) : 0.0;


        // Spacing check: require Ypmax and Yhmin to be at least shellGap (+thickness) inside previous
        if ((prev_Ypmax - Ypmax) < shellGap + shell_thickness) continue;
        if ((prev_Yhmin - Yhmin) < (shellGap + shell_thickness)) continue;
        if (Yhmin < Rdumbell + shellGap) break; // Stop if hyperboloid tip is too close to dumbell

        // --- Build layers for paraboloid ---
        for (int layer = 0; layer < 3; ++layer) {
            std::vector<double> xP, rInnerP, rOuterP;
            for (int j = 0; j <= N; ++j) {
                double x = Xpmin + (Xpmax - Xpmin) * j / N;
                double y = std::sqrt(p * (2 * x + p));
                xP.push_back(x); // x is along the optical axis
                rInnerP.push_back(y + layer_inner[layer]);
                rOuterP.push_back(y + layer_outer[layer]);
            }
            std::ostringstream solidName;
            solidName << "WolterI_Primary_" << shellIdx << "_" << layer_name[layer];
            G4Polycone* shell = new G4Polycone(
                solidName.str().c_str(), 0, 2*M_PI, N+1,
                xP.data(), rInnerP.data(), rOuterP.data()
            );
            std::ostringstream lvName;
            lvName << "WolterI_PrimaryLV_" << shellIdx << "_" << layer_name[layer];
            G4LogicalVolume* logicShell = new G4LogicalVolume(shell, layer_mat[layer], lvName.str().c_str());

            // Rotate by +90 deg about Y so x (optical axis) aligns with z
            G4RotationMatrix* rotY90 = new G4RotationMatrix();
            rotY90->rotateY(-90.*CLHEP::deg);

            // Place at origin (no offset between para and hyper)
            new G4PVPlacement(rotY90, G4ThreeVector(-FL, 0, 0), logicShell, solidName.str().c_str(), logicWorld, false, shellIdx*6+layer, true);

            G4Colour col;
            if (layer == 0)      col = G4Colour(1.0, 0.84, 0.0); // Gold/yellow
            else if (layer == 1) col = G4Colour(1.0, 0.84, 0.0);   // Blue (epoxy)
            else                 col = G4Colour(0.7, 0.7, 0.7);   // Grey (aluminium)
            G4VisAttributes* shellVis = new G4VisAttributes(col);
            shellVis->SetForceSolid(true);
            logicShell->SetVisAttributes(shellVis);
        }

        // --- Build layers for hyperboloid ---
        for (int layer = 0; layer < 3; ++layer) {
            std::vector<double> xH, rInnerH, rOuterH;
            for (int j = 0; j <= N; ++j) {
                double x = Xhmin + (Xhmax - Xhmin) * j / N;
                double arg = ((x - c)*(x - c)) / (a*a) - 1.0;
                if (arg < 0) continue;
                double y = b * std::sqrt(arg);
                xH.push_back(x); // x is along the optical axis
                rInnerH.push_back(y + layer_inner[layer]);
                rOuterH.push_back(y + layer_outer[layer]);
            }
            if (xH.size() < 2) continue;
            std::ostringstream solidName;
            solidName << "WolterI_Secondary_" << shellIdx << "_" << layer_name[layer];
            G4Polycone* shell = new G4Polycone(
                solidName.str().c_str(), 0, 2*M_PI, xH.size(),
                xH.data(), rInnerH.data(), rOuterH.data()
            );
            std::ostringstream lvName;
            lvName << "WolterI_SecondaryLV_" << shellIdx << "_" << layer_name[layer];
            G4LogicalVolume* logicShell = new G4LogicalVolume(shell, layer_mat[layer], lvName.str().c_str());

            // Rotate by +90 deg about Y so x (optical axis) aligns with z
            G4RotationMatrix* rotY90 = new G4RotationMatrix();
            rotY90->rotateY(-90.*CLHEP::deg);

            // Place at origin (no offset between para and hyper)
            new G4PVPlacement(rotY90, G4ThreeVector(-FL, 0, 0), logicShell, solidName.str().c_str(), logicWorld, false, shellIdx*6+3+layer, true);

            G4Colour col;
            if (layer == 0)      col = G4Colour(1.0, 0.84, 0.0); // Gold/yellow
            else if (layer == 1) col = G4Colour(1.0, 0.84, 0.0);   // Blue (epoxy)
            else                 col = G4Colour(0.7, 0.7, 0.7);   // Grey (aluminium)
            G4VisAttributes* shellVis = new G4VisAttributes(col);
            shellVis->SetForceSolid(true);
            logicShell->SetVisAttributes(shellVis);
        }

        // --- Print info for this shell, now including the min_Rc ---
        G4cout << "Wolter-I shell " << shellIdx
               << ": intersection (x=" << x_int/CLHEP::mm << " mm, y=" << y_int/CLHEP::mm << " mm), "
               << "Ypmax=" << Ypmax/CLHEP::mm << " mm, Yhmin=" << Yhmin/CLHEP::mm << " mm, "
               << "min_Rc_H=" << min_Rc_hyperboloid << " m"
               << G4endl;

        // Prepare for next shell
        prev_Ypmax = Ypmax;
        prev_Yhmin = Yhmin;
        shellIdx++;
    }
    G4cout << "BuildNestedWolterI CALLED" << G4endl;
}