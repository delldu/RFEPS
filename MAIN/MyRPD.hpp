#pragma once
// RVD.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <set>
#include <omp.h> 
#include <cstdio>
#include <cstring>
#include <queue>

#include"MyHalfEdgeModel.hpp"
#include"PlaneCut.hpp"
#include"Knn.hpp"
#include"MyPointCloudModel.hpp"
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Weighted_point_3.h>
#include <CGAL/bounding_box.h>

typedef CGAL::Simple_cartesian<double> K_T;
typedef CGAL::Polyhedron_3<K_T> Polyhedron;

// for power d
typedef CGAL::Exact_predicates_inexact_constructions_kernel K_P;
typedef CGAL::Regular_triangulation_3<K_P> Regular_triangulation;
typedef K_P::Point_3 Point_P;
typedef K_P::Weighted_point_3 Wp;
typedef CGAL::Regular_triangulation_3<K_P> Rt;


using namespace std;

#define gamma 0.00000000001

double FeatureWeight = 0.015, noFeatureWeight = 0.005;

struct MyPoint
{
    MyPoint(Eigen::Vector3d a)
    {
        p = a;
    }

    MyPoint(double a, double b, double c)
    {
        p.x() = a;
        p.y() = b;
        p.z() = c;
    }
    Eigen::Vector3d p;

    bool operator<(const MyPoint& a) const
    {
        double dis = (p - a.p).norm();
        if (dis < gamma)
            return false;

        if ((p.x() - a.p.x()) < 0.00000000001 && (p.x() - a.p.x()) > -0.00000000001) {
            if ((p.y() - a.p.y()) < 0.00000000001 && (p.y() - a.p.y()) > -0.00000000001)
                return (p.z() < a.p.z());
            return (p.y() < a.p.y());
        }

        return (p.x() < a.p.x());
    }

    bool operator==(const MyPoint& a) const
    {
        if ((p.x() - a.p.x()) < 0.00000000001 && (p.x() - a.p.x()) > -0.00000000001) {
            if ((p.y() - a.p.y()) < 0.00000000001 && (p.y() - a.p.y()) > -0.00000000001) {
                if ((p.z() - a.p.z()) < 0.00000000001 && (p.z() - a.p.z()) > -0.00000000001)
                    return 1;
            }
        }
        return 0;
    }
};

struct MyFace
{
    MyFace(Eigen::Vector3i a)
    {
        p = a;
    }
    MyFace(int a, int b, int c)
    {
        p.x() = a;
        p.y() = b;
        p.z() = c;
    }

    bool operator<(const MyFace& a) const
    {
        if (p.x() == a.p.x()) {
            if (p.y() == a.p.y())
                return p.z() > a.p.z();
            return p.y() > a.p.y();
        }
        return p.x() > a.p.x();
    }

    Eigen::Vector3i p;
};


vector<Regular_triangulation::Weighted_point> wpoints;

void add_point(double x, double y, double z, double w) {
    wpoints.push_back(Wp(Point_P(x, y, z), w));
};

void Comput_RPD(string modelpath, string modelName)
{
    string rpd_output = ".";

    wpoints.clear();
    vector<bool> IsFeature;
    vector<double> Weight;

    map<pair<MyPoint, MyPoint>, int> RVD;
    map<pair<int, int>, bool> FlagOf2Points;
    map<pair<int, int>, int> linefix;
    vector<Eigen::Vector3d> VersPC_ori, Normal_ori;
    vector<Eigen::Vector3d> VersPC;
    map<MyFace, int> NewFaces;
    map<int, PlaneCut*> PCs;
    map<int, set<MyPoint>> RVDpoints;


    MyHalfEdgeModel* PoissonModel = new MyHalfEdgeModel();
    PoissonModel->ReadObjFile((rpd_output + "/model_poisson_"+ modelName +".obj").c_str());

    double radis = 0;
    std::cout << "Comput_RPD: Loading " << rpd_output  + "/radis_" + modelName + ".txt" << " ..." << std::endl;
    ifstream inRnum(rpd_output  + "/radis_" + modelName + ".txt");
    inRnum >> radis;
    inRnum.close();
    cout << radis << endl;

    FeatureWeight = radis*1.0;
    noFeatureWeight = radis / 3.0;

    std::cout << "Comput_RPD: Loading " << rpd_output  + "/FeaturePointNum_" + modelName + ".txt" << " ..." << std::endl;
    ifstream inFnum(rpd_output  + "/FeaturePointNum_" + modelName + ".txt");
    // ==> IsFeature

    std::cout << "Comput_RPD: Loading " << rpd_output  + "/PoissonPoints_qc_" + modelName + ".xyz" << " ..." << std::endl;
    ifstream inNewPs(rpd_output  + "/PoissonPoints_qc_" + modelName + ".xyz");
    // ==> VersPC

    std::cout << "Comput_RPD: Loading " << rpd_output  + "/OriPoints_qc_" + modelName + ".xyz" << " ..." << std::endl;
    ifstream inOriPs(rpd_output  + "/OriPoints_qc_" + modelName + ".xyz");
    // ==> VersPC_ori, Normal_ori

    int n = 0;
    double x11, y11, z11;
    while (inNewPs >> x11 >> y11 >> z11) {
        n++;
        bool isf;
        inFnum >> isf;

        IsFeature.push_back(isf);
        Eigen::Vector3d NewP(x11, y11, z11);
        VersPC.push_back(NewP);

        inOriPs >> x11 >> y11 >> z11;
        Eigen::Vector3d NewP2(x11, y11, z11);
        VersPC_ori.push_back(NewP2);

        inOriPs >> x11 >> y11 >> z11;
        Eigen::Vector3d normal(x11, y11, z11);
        Normal_ori.push_back(normal);
    }

    map<MyPoint, int> Point2ID;
    vector<vector<int>> neighboor;

    Knn KnnPoisson(rpd_output + "/knn50_poisson_" + modelName + ".txt", n, 50, false);

    for (int i = 0; i < n; i++) {
        if (IsFeature[i]) {
            add_point(VersPC[i].x(), VersPC[i].y(), VersPC[i].z(), FeatureWeight * FeatureWeight);
            Weight.push_back(FeatureWeight);
        } else {
            add_point(VersPC[i].x(), VersPC[i].y(), VersPC[i].z(), noFeatureWeight * noFeatureWeight);
            Weight.push_back(noFeatureWeight);
        }
        Point2ID[MyPoint(VersPC[i])] = i;
    }
    Regular_triangulation rt(wpoints.begin(), wpoints.end());
    rt.is_valid();

    // cout << "make Regular_triangulation .\n";
    for (int i = 0; i < n; i++) {
        vector<int> tmp;
        neighboor.push_back(tmp);
    }

    for (const Rt::Vertex_handle vh : rt.finite_vertex_handles()) {
        std::vector<Rt::Vertex_handle> f_vertices;

        rt.finite_adjacent_vertices(vh, std::back_inserter(f_vertices));
        vector<int> nb_tmps;
        for (auto nb : f_vertices) {
            if (Point2ID.find(MyPoint(nb->point().x(), nb->point().y(), nb->point().z())) == Point2ID.end())
                std::cout << "ERROR !" << std::endl;
            nb_tmps.push_back(Point2ID[MyPoint(nb->point().x(), nb->point().y(), nb->point().z())]);
        }
        if (Point2ID.find(MyPoint(vh->point().x(), vh->point().y(), vh->point().z())) == Point2ID.end())
            std::cout << "ERROR !" << std::endl;

        neighboor[Point2ID[MyPoint(vh->point().x(), vh->point().y(), vh->point().z())]] = nb_tmps;
    }


    auto VersPoisson = PoissonModel->GetVertices();
    auto FacesPoisson = PoissonModel->GetFaces();

    omp_set_num_threads(16);
#pragma omp parallel for //schedule(dynamic, 20)
    for (int ii = 0; ii < n; ii++) {
        PlaneCut* PC = new PlaneCut(VersPC[ii]);

        map<int, int> opposide;
        map<MyPoint, vector<int>> point2edge;

        int planeNum = 0;

        //for (int jj = KnnPC.neighboor[ii].size() - 1; jj >= 0; jj--)
        for (int jj = 0; jj < neighboor[ii].size(); jj++) {
            //cout << " " << jj << " ";
            auto p = neighboor[ii][jj];

            auto aa = VersPC[ii];
            auto bb = VersPC[p];
            auto r1 = Weight[ii], r2 = Weight[p];

            Eigen::Vector3d MidPoint;
            double lambda = ((r1 * r1 - r2 * r2) / ((bb - aa).norm() * (bb - aa).norm()) + 1.0) / 2.0;
            MidPoint = (1 - lambda) * aa + lambda * bb;

            K::Point_3 point1(MidPoint.x(), MidPoint.y(), MidPoint.z());
            MidPoint = VersPC[p] - VersPC[ii];

            MidPoint.normalize();
            K::Direction_3 dir(MidPoint.x(), MidPoint.y(), MidPoint.z());

            Plane plane(point1, dir);
            bool ifcut = PC->CutByPlane(plane);

            if (ifcut) {
                opposide[planeNum] = p; planeNum++;
            }
            if (IsFeature[ii] && ifcut) {
                linefix[make_pair(ii, p)] = 1;
            }
        }

        PCs[ii] = PC;
        vector<bool> FlagPlane;
        for (int pl = 0; pl < PC->Planes.size(); pl++) {
            auto e = make_pair(min(ii, opposide[pl]), max(ii, opposide[pl]));
            if (FlagOf2Points.find(e) == FlagOf2Points.end()) {
                FlagOf2Points[e] = true;
                FlagPlane.push_back(1);
            } else {
                FlagPlane.push_back(0);
            }
        }

        set<int> CloseFaces;
        for (auto p : KnnPoisson.neighboor[ii]) {
            vector<int> PFaces = PoissonModel->GetFacesByPoint(p);
            for (auto f : PFaces) {
                CloseFaces.insert(f);
            }
        }

        vector<vector<Eigen::Vector3d>> CuttedF;
        vector<bool> aliveF;

        map<MyPoint, int> PointType;
        for (auto f : CloseFaces) {
            bool cuted = 0;

            auto P1 = VersPoisson[FacesPoisson[f].x()];
            auto P2 = VersPoisson[FacesPoisson[f].y()];
            auto P3 = VersPoisson[FacesPoisson[f].z()];
            K::Point_3 P1_Cgal(P1.x(), P1.y(), P1.z());
            K::Point_3 P2_Cgal(P2.x(), P2.y(), P2.z());
            K::Point_3 P3_Cgal(P3.x(), P3.y(), P3.z());
            MyPoint PP1(P1), PP2(P2), PP3(P3);
            bool vp1 = 0, vp2 = 0, vp3 = 0;
            PointType[PP1] = 0; PointType[PP2] = 0; PointType[PP3] = 0;
            for (int pl = 0; pl < PC->Planes.size(); pl++) {
                if (! FlagPlane[pl])
                    continue;

                auto plane = PC->Planes[pl];
                if (plane.has_on_positive_side(P1_Cgal)) {
                    vp1 = 1;
                    PointType[PP1] = 1;
                }
                if (plane.has_on_positive_side(P2_Cgal)) {
                    vp2 = 1;
                    PointType[PP2] = 1;
                }
                if (plane.has_on_positive_side(P3_Cgal)) {
                    vp3 = 1;
                    PointType[PP3] = 1;
                }
                double f1, f2, f3;
                f1 = plane.a() * P1.x() + plane.b() * P1.y() + plane.c() * P1.z() + plane.d();
                f2 = plane.a() * P2.x() + plane.b() * P2.y() + plane.c() * P2.z() + plane.d();
                f3 = plane.a() * P3.x() + plane.b() * P3.y() + plane.c() * P3.z() + plane.d();

                if ((f1 > 0 && f2 < 0) || (f1 < 0 && f2 > 0)) {
                    cuted = 1;
                }
                if ((f2 > 0 && f3 < 0) || (f2 < 0 && f3 > 0)) {
                    cuted = 1; //cuttedFp.push_back(opposide[plane]);
                }
                if ((f1 > 0 && f3 < 0) || (f1 < 0 && f3 > 0)) {
                    cuted = 1; //cuttedFp.push_back(opposide[plane]);
                }
            }

            if (cuted) {
                vector<Eigen::Vector3d> thisF;
                thisF.push_back(VersPoisson[FacesPoisson[f].x()]);
                thisF.push_back(VersPoisson[FacesPoisson[f].y()]);
                thisF.push_back(VersPoisson[FacesPoisson[f].z()]);
                CuttedF.push_back(thisF);

                aliveF.push_back(1);
            }
        }

        for (int j = 0; j < CuttedF.size(); j++) {
            auto f = CuttedF[j];
            if (!aliveF[j])
                continue;

            vector<Eigen::Vector3d> newF;

            bool fd = 0;
            for (int pl = 0; pl < PC->Planes.size(); pl++) {
                auto plane = PC->Planes[pl];
                if (fd)
                    break;

                vector<Eigen::Vector3d> newF_tmp;
                for (int i = 0; i < f.size(); i++) {
                    Eigen::Vector3d P1, P2;
                    P1 = f[i];
                    P2 = (i == f.size() - 1)? f[0] : f[i + 1];

                    double f1, f2;
                    f1 = plane.a() * P1.x() + plane.b() * P1.y() + plane.c() * P1.z() + plane.d();
                    f2 = plane.a() * P2.x() + plane.b() * P2.y() + plane.c() * P2.z() + plane.d();
                    if (fabs(f1) < gamma) {
                        newF_tmp.push_back(P1);
                        MyPoint PP1(P1);
                        point2edge[PP1].push_back(opposide[pl]);
                        continue;
                    }
                    if (fabs(f2) < gamma) {
                        newF_tmp.push_back(P1);
                        MyPoint PP1(P1);
                        point2edge[P2].push_back(opposide[pl]);
                        continue;
                    }

                    if ((f1 > 0 && f2 < 0) || (f1 < 0 && f2 > 0)) {
                        Eigen::Vector3d NewPoint;
                        f1 = fabs(f1); f2 = fabs(f2);
                        NewPoint.x() = P1.x() + (P2.x() - P1.x()) * (f1 / (f1 + f2));
                        NewPoint.y() = P1.y() + (P2.y() - P1.y()) * (f1 / (f1 + f2));
                        NewPoint.z() = P1.z() + (P2.z() - P1.z()) * (f1 / (f1 + f2));
                        MyPoint PP1(NewPoint);
                        newF_tmp.push_back(P1);
                        newF_tmp.push_back(NewPoint);

                        point2edge[PP1].push_back(opposide[pl]);
                        PointType[NewPoint] = -1;
                        fd = 1;
                    } else {
                        newF_tmp.push_back(P1);
                    }
                }
                if (fd) {
                    newF = newF_tmp;
                }
            }

            if (fd == 0) {
                bool isInside = 0;// for test
                bool onEdge = 1;
                double maxFo = -99999999.0;
                for (int i = 0; i < f.size(); i++) {
                    if (PointType[f[i]] == 0)
                        isInside = 1;

                    Eigen::Vector3d P1;
                    P1 = f[i];
                    for (int pl = 0; pl < PC->Planes.size(); pl++) {
                        auto plane = PC->Planes[pl];

                        double f1;
                        f1 = plane.a() * P1.x() + plane.b() * P1.y() + plane.c() * P1.z() + plane.d();
                        if (f1 > maxFo)
                            maxFo = f1;
                    }
                }
                if (maxFo < gamma)
                    isInside = 1;

                if (isInside) {
                    vector < bool > Nps;
                    vector < Eigen::Vector3d > Newps;
                    for (int i = 0; i < f.size(); i++) {
                        Eigen::Vector3d P1;
                        P1 = f[i];
                        double maxF = -99999999.0;
                        for (int pl = 0; pl < PC->Planes.size(); pl++) {
                            auto plane = PC->Planes[pl];

                            double f1, f2;
                            f1 = plane.a() * P1.x() + plane.b() * P1.y() + plane.c() * P1.z() + plane.d();
                            maxF = max(maxF, f1);
                        }

                        if (fabs(maxF) < gamma) {
                            Nps.push_back(1);
                            Newps.push_back(f[i]);
                        } else {
                            Nps.push_back(0);
                        }
                    }
                    if (Newps.size() < 2)
                        continue;

                    vector<Eigen::Vector3d> Drawedpoints;
                    for (int i = 0; i < f.size(); i++) {
                        Eigen::Vector3d P1, P2, Pmid;
                        P1 = f[i];
                        P2 = (i == f.size() - 1)? f[0] : f[i + 1];

                        bool V1, V2;
                        V1 = Nps[i];
                        V2 = (i == f.size() - 1)? Nps[0] : Nps[i + 1];

                        bool doublecheck = 0;
                        Pmid.x() = P1.x() + (P2.x() - P1.x()) / 2;
                        Pmid.y() = P1.y() + (P2.y() - P1.y()) / 2;
                        Pmid.z() = P1.z() + (P2.z() - P1.z()) / 2;
                        double maxF = -99999999.0;
                        for (int pl = 0; pl < PC->Planes.size(); pl++) {
                            auto plane = PC->Planes[pl];

                            double f1, f2;
                            f1 = plane.a() * Pmid.x() + plane.b() * Pmid.y() + plane.c() * Pmid.z() + plane.d();
                            maxF = max(maxF, f1);

                        }
                        doublecheck = (fabs(maxF) < gamma)? 0 : 1;

                        if (V1 && V2 && !doublecheck) {
                            Drawedpoints.push_back(P1);
                            Drawedpoints.push_back(P2);

                            MyPoint p1(P1), p2(P2);
                            RVDpoints[ii].insert(p1);
                            RVDpoints[ii].insert(p2);

                            if (p2 < p1) {
                                if (RVD.find(make_pair(p1, p2)) == RVD.end()) {
                                    RVD[make_pair(p1, p2)] = ii;
                                } else {
                                    int lst = RVD[make_pair(p1, p2)];
                                }
                            } else {
                                if (RVD.find(make_pair(p2, p1)) == RVD.end()) {
                                    RVD[make_pair(p2, p1)] = ii;
                                } else {
                                    int lst = RVD[make_pair(p2, p1)];
                                }
                            }
                        }
                    }
                    map<int, int> cnt;
                    for (auto mp : Drawedpoints) {
                        auto P1 = mp;
                        double maxF = -999999.0;
                        for (int pl = 0; pl < PC->Planes.size(); pl++) {
                            auto plane = PC->Planes[pl];
                            double f1;
                            f1 = plane.a() * P1.x() + plane.b() * P1.y() + plane.c() * P1.z() + plane.d();
                            if (fabs(f1) < gamma) {
                                int oppoP = opposide[pl];
                                if (cnt.find(oppoP) == cnt.end()) {
                                    cnt[oppoP] = 1;
                                } else {
                                    cnt[oppoP]++;
                                }
                            }
                        }
                    }
                } else {
                    aliveF[j] = 0;
                }
            } else {
                vector<Eigen::Vector3d> newF1, newF2;
                bool fdd = 0;
                for (int i = 0; i < newF.size(); i++) {
                    if (PointType[newF[i]] == -1) {
                        PointType[newF[i]] = -2;
                        if (fdd == 0) {
                            newF1.push_back(newF[i]); newF2.push_back(newF[i]);
                            fdd = 1;
                            continue;
                        } else {
                            newF1.push_back(newF[i]);
                            newF2.push_back(newF[i]);
                            fdd = 0;
                            continue;
                        }
                    }
                    if (fdd == 0) {
                        newF1.push_back(newF[i]); 
                    } else {
                        newF2.push_back(newF[i]);
                    }
                }
                aliveF[j] = 0;
                aliveF.push_back(1);
                aliveF.push_back(1);
                CuttedF.push_back(newF1);
                CuttedF.push_back(newF2);
            }
        }

        for (auto rvdp : RVDpoints[ii]) {
            auto P1 = rvdp.p;
            double maxF = -999999.0;
            for (int pl = 0; pl < PC->Planes.size() - 1; pl++) {
                for (int pl2 = pl + 1; pl2 < PC->Planes.size(); pl2++) {
                    auto plane1 = PC->Planes[pl];
                    auto plane2 = PC->Planes[pl2];
                    double f1, f2;

                    f1 = plane1.a() * P1.x() + plane1.b() * P1.y() + plane1.c() * P1.z() + plane1.d();
                    f2 = plane2.a() * P1.x() + plane2.b() * P1.y() + plane2.c() * P1.z() + plane2.d();
                    if (fabs(f1) < gamma && fabs(f2) < gamma) {
                        int aa, bb, cc;

                        aa = min(ii, min(opposide[pl], opposide[pl2]));
                        cc = max(ii, max(opposide[pl], opposide[pl2]));
                        if (ii != aa && ii != cc)
                            bb = ii;
                        if (opposide[pl] != aa && opposide[pl] != cc)
                            bb = opposide[pl];
                        if (opposide[pl2] != aa && opposide[pl2] != cc)
                            bb = opposide[pl2];

                        MyFace ff(aa, bb, cc);
                        NewFaces[ff] = ii;
                    }
                }
            }
        }
        //break;
    }


    map<int, bool> ifInsideOtherRVD;
    for (int ii = 0; ii < n; ii++) {
        auto PC = PCs[ii];

        ifInsideOtherRVD[ii] = 0;
        bool flagRvdp = 0;
        for (auto rvdp : RVDpoints[ii]) {
            bool flagii = 0;
            for (int jj = 0; jj < neighboor[ii].size(); jj++) {
                auto p = neighboor[ii][jj];

                if (PCs.find(p) == PCs.end())
                    continue;

                if (PCs[p]->Planes.empty())
                    continue;
                auto P1 = rvdp.p;

                flagii = 0;
                for (int pl = 0; pl < PCs[p]->Planes.size(); pl++) {
                    auto plane = PCs[p]->Planes[pl];
                    double f1;
                    f1 = plane.a() * P1.x() + plane.b() * P1.y() + plane.c() * P1.z() + plane.d();
                    if (f1 >= 0 || fabs(f1) <= gamma) {
                        flagii = 1;
                        break;
                    }
                }

                if (flagii == 0) {
                    break;
                }
            }
            if (flagii == 1) {
                flagRvdp = 1;
                break;
            }
        }

        if (!flagRvdp)
            ifInsideOtherRVD[ii] = 1;
    }


    std::cout << "Comput_RPD: Saving " << rpd_output + "/RVD_" + modelName + ".obj" << " ..." << std::endl;
    ofstream outRVD(rpd_output + "/RVD_" + modelName + ".obj");
    outRVD.precision(15);
    outRVD.flags(ios::left | ios::fixed);
    outRVD.fill('0');

    map<MyPoint, int> Point2IDD;
    int pid = 0;
    for (auto mp : RVD) {
        MyPoint P1 = mp.first.first;
        MyPoint P2 = mp.first.second;

        if (Point2IDD.find(P1) == Point2IDD.end()) {
            pid++;
            Point2IDD[P1] = pid;
            outRVD << "v " << P1.p.transpose() << "\n";
        }
        if (Point2IDD.find(P2) == Point2IDD.end()) {
            pid++;
            Point2IDD[P2] = pid;
            outRVD << "v " << P2.p.transpose() << "\n";
        }
        outRVD << "l " << Point2IDD[P1] << " " << Point2IDD[P2] << "\n";
    }
    outRVD.close();

    map<pair<int, int>, int> DegreeOfEdge;
    cout << NewFaces.size() << endl;
    for (auto f : NewFaces) {
        if (ifInsideOtherRVD[f.second] == 1)
            continue;

        int aa, bb, cc;
        aa = f.first.p.x();
        bb = f.first.p.y();
        cc = f.first.p.z();

        if (DegreeOfEdge.find(make_pair(min(aa, bb), max(aa, bb))) == DegreeOfEdge.end()) {
            DegreeOfEdge[make_pair(min(aa, bb), max(aa, bb))] = 0;
        }
        if (DegreeOfEdge.find(make_pair(min(aa, cc), max(aa, cc))) == DegreeOfEdge.end()) {
            DegreeOfEdge[make_pair(min(aa, cc), max(aa, cc))] = 0;
        }
        if (DegreeOfEdge.find(make_pair(min(cc, bb), max(cc, bb))) == DegreeOfEdge.end()) {
            DegreeOfEdge[make_pair(min(cc, bb), max(cc, bb))] = 0;
        }
        DegreeOfEdge[make_pair(min(aa, bb), max(aa, bb))]++;
        DegreeOfEdge[make_pair(min(aa, cc), max(aa, cc))]++;
        DegreeOfEdge[make_pair(min(cc, bb), max(cc, bb))]++;
    }

    while (true) {
        bool vvv = 0;
        for (auto f : NewFaces) {
            if (f.second == -1)
                continue;

            if (ifInsideOtherRVD[f.second] == 1)
                continue;

            int aa, bb, cc;
            aa = f.first.p.x();
            bb = f.first.p.y();
            cc = f.first.p.z();

            if (DegreeOfEdge[make_pair(min(aa, bb), max(aa, bb))] < 2 || DegreeOfEdge[make_pair(min(aa, cc), max(aa, cc))] < 2 || DegreeOfEdge[make_pair(min(cc, bb), max(cc, bb))] < 2)
            {
                if (DegreeOfEdge[make_pair(min(aa, bb), max(aa, bb))] > 2 || DegreeOfEdge[make_pair(min(aa, cc), max(aa, cc))] > 2 || DegreeOfEdge[make_pair(min(cc, bb), max(cc, bb))] > 2)
                {
                    DegreeOfEdge[make_pair(min(aa, bb), max(aa, bb))]--;
                    DegreeOfEdge[make_pair(min(aa, cc), max(aa, cc))]--;
                    DegreeOfEdge[make_pair(min(cc, bb), max(cc, bb))]--;
                    MyFace ff(aa, bb, cc);
                    NewFaces[ff] = -1;
                    vvv = 1;
                }
            }
        }
        if (! vvv) {
            break;
        }
    }
    vector<Eigen::Vector3i> RemeshFs;
    int fid = 0;
    map<pair<int, int>, vector<int>> Edge2Faceid;
    vector<bool> FaceFlag;
    for (auto f : NewFaces) {
        if (f.second == -1)
            continue;

        if (ifInsideOtherRVD[f.second] == 1)
            continue;

        Edge2Faceid[make_pair(min(f.first.p.x(), f.first.p.y()), max(f.first.p.x(), f.first.p.y()))].push_back(fid);
        Edge2Faceid[make_pair(min(f.first.p.z(), f.first.p.y()), max(f.first.p.z(), f.first.p.y()))].push_back(fid);
        Edge2Faceid[make_pair(min(f.first.p.x(), f.first.p.z()), max(f.first.p.x(), f.first.p.z()))].push_back(fid);

        FaceFlag.push_back(0);
        RemeshFs.push_back(Eigen::Vector3i(f.first.p.x(), f.first.p.y(), f.first.p.z()));
        fid++;
    }

    // fix normal
    int NormalRightFaceID = -1;
    for (int i = 0; i < RemeshFs.size(); i++) {
        auto f = RemeshFs[i];
        if (!IsFeature[f.x()] && !IsFeature[f.y()] && !IsFeature[f.z()]) {
            NormalRightFaceID = i;

            auto nor1 = (VersPC_ori[f.y()] - VersPC_ori[f.x()]).cross(VersPC_ori[f.z()] - VersPC_ori[f.y()]);
            nor1.normalize();
            auto nor2 = (VersPC_ori[f.x()] - VersPC_ori[f.y()]).cross(VersPC_ori[f.z()] - VersPC_ori[f.x()]);
            nor2.normalize();

            Eigen::Vector3d nor3 = (Normal_ori[f.x()].normalized() + Normal_ori[f.y()].normalized() + Normal_ori[f.z()].normalized()) / 3.0;
            nor3.normalize();
            double dis1 = (nor3 - nor1).norm();
            double dis2 = (nor3 - nor2).norm();

            if (dis2 < dis1) {
                int tmpp = RemeshFs[i].x();
                RemeshFs[i].x() = RemeshFs[i].y();
                RemeshFs[i].y() = tmpp;
            }
            FaceFlag[i] = 1;

            break;
        }
    }

    queue<int> FaceQue; FaceQue.push(NormalRightFaceID);
    while (!FaceQue.empty()) {
        int faceid = FaceQue.front();
        FaceQue.pop();

        auto f = RemeshFs[faceid];
        auto FaceSet1 = Edge2Faceid[make_pair(min(f.x(), f.y()), max(f.x(), f.y()))];
        int aa = f.x(), bb = f.y();
        for (auto ff : FaceSet1) {
            if (FaceFlag[ff] != 0)
                continue;

            // FaceFlag[ff] == 0
            FaceQue.push(ff);
            FaceFlag[ff] = 1;
            auto tf = RemeshFs[ff];
            if (tf.x() == aa && tf.y() == bb) {
                int tmpf = RemeshFs[ff].y();
                RemeshFs[ff].y() = RemeshFs[ff].x();
                RemeshFs[ff].x() = tmpf;
                continue;
            }

            if (tf.y() == aa && tf.z() == bb) {
                int tmpf = RemeshFs[ff].y();
                RemeshFs[ff].y() = RemeshFs[ff].z();
                RemeshFs[ff].z() = tmpf;
                continue;
            }

            if (tf.z() == aa && tf.x() == bb) {
                int tmpf = RemeshFs[ff].z();
                RemeshFs[ff].z() = RemeshFs[ff].x();
                RemeshFs[ff].x() = tmpf;
                continue;
            }
        }

        // second edge 
        auto FaceSet2 = Edge2Faceid[make_pair(min(f.z(), f.y()), max(f.z(), f.y()))];
        aa = f.y(); bb = f.z();
        for (auto ff : FaceSet2) {
            if (FaceFlag[ff] != 0)
                continue;

            // FaceFlag[ff] == 0
            FaceFlag[ff] = 1; FaceQue.push(ff);
            auto tf = RemeshFs[ff];
            if (tf.x() == aa && tf.y() == bb) {
                int tmpf = RemeshFs[ff].y();
                RemeshFs[ff].y() = RemeshFs[ff].x();
                RemeshFs[ff].x() = tmpf;
                continue;
            }

            if (tf.y() == aa && tf.z() == bb) {
                int tmpf = RemeshFs[ff].y();
                RemeshFs[ff].y() = RemeshFs[ff].z();
                RemeshFs[ff].z() = tmpf;
                continue;
            }

            if (tf.z() == aa && tf.x() == bb) {
                int tmpf = RemeshFs[ff].z();
                RemeshFs[ff].z() = RemeshFs[ff].x();
                RemeshFs[ff].x() = tmpf;
                continue;
            }
        }

        // third edge
        auto FaceSet3 = Edge2Faceid[make_pair(min(f.x(), f.z()), max(f.x(), f.z()))];
        aa = f.z(); bb = f.x();
        for (auto ff : FaceSet3) {
            if (FaceFlag[ff] != 0)
                continue;

            // FaceFlag[ff] == 0
            FaceFlag[ff] = 1; FaceQue.push(ff);
            auto tf = RemeshFs[ff];
            if (tf.x() == aa && tf.y() == bb) {
                int tmpf = RemeshFs[ff].y();
                RemeshFs[ff].y() = RemeshFs[ff].x();
                RemeshFs[ff].x() = tmpf;
                continue;
            }

            if (tf.y() == aa && tf.z() == bb) {
                int tmpf = RemeshFs[ff].y();
                RemeshFs[ff].y() = RemeshFs[ff].z();
                RemeshFs[ff].z() = tmpf;
                continue;
            }

            if (tf.z() == aa && tf.x() == bb) {
                int tmpf = RemeshFs[ff].z();
                RemeshFs[ff].z() = RemeshFs[ff].x();
                RemeshFs[ff].x() = tmpf;
                continue;
            }
        }
    }

    // delete nonf
    map<pair<int, int>, int> EdgeCnt;
    for (auto f : RemeshFs) {
        int aa = f.x();
        int bb = f.y();
        int cc = f.z();
        auto e = make_pair(min(aa, bb), max(aa, bb));
        if (EdgeCnt.find(e) == EdgeCnt.end()) {
            EdgeCnt[e] = 1;
        } else {
            EdgeCnt[e]++;
        }
        e = make_pair(min(aa, cc), max(aa, cc));
        if (EdgeCnt.find(e) == EdgeCnt.end()) {
            EdgeCnt[e] = 1;
        } else {
            EdgeCnt[e]++;
        }
        e = make_pair(min(cc, bb), max(cc, bb));
        if (EdgeCnt.find(e) == EdgeCnt.end()) {
            EdgeCnt[e] = 1;
        } else {
            EdgeCnt[e]++;
        }
    }

    std::cout << "Comput_RPD: Saving " << rpd_output + "/Remesh_" + modelName + ".obj" << " ..." << std::endl;
    ofstream outRemesh(rpd_output + "/Remesh_" + modelName + ".obj");
    outRemesh.precision(15);
    outRemesh.flags(ios::left | ios::fixed);
    outRemesh.fill('0');
    for (auto p : VersPC_ori)
        outRemesh << "v " << p.transpose() << endl;

    for (auto f : RemeshFs) {
        int aa = f.x();
        int bb = f.y();
        int cc = f.z();

        if (EdgeCnt[make_pair(min(aa, bb), max(aa, bb))] == 1)
            continue;

        if (EdgeCnt[make_pair(min(cc, bb), max(cc, bb))] == 1)
            continue;

        if (EdgeCnt[make_pair(min(aa, cc), max(aa, cc))] == 1)
            continue;

        outRemesh << "f " << f.x() + 1 << " " << f.y() + 1 << " " << f.z() + 1 << endl;
    }
    outRemesh.close();

    std::cout << "Comput_RPD: Saving " << rpd_output +  "/Edges_" + modelName + ".obj" << " ..." << std::endl;
    ofstream outEdge(rpd_output +  "/Edges_" + modelName + ".obj");

    outEdge.precision(15);
    outEdge.flags(ios::left | ios::fixed);
    outEdge.fill('0');

    set<pair<int, int>> edges;
    for (auto p : VersPC_ori) {
        outEdge << "v " << p.transpose() << endl;
    }

    for (auto f : RemeshFs) {
        edges.insert(make_pair(min(f.x(), f.y()), max(f.x(), f.y())));
        edges.insert(make_pair(min(f.z(), f.y()), max(f.z(), f.y())));
        edges.insert(make_pair(min(f.x(), f.z()), max(f.x(), f.z())));
    }

    for (auto e : edges) {
        outEdge << "l " << e.first + 1 << " " << e.second + 1 << "\n";
    }
    outEdge.close();

    std::cout << "Comput_RPD: Saving " << rpd_output + "/FeatureLine_" + modelName + ".obj" << " ..." << std::endl;
    ofstream outFeatureLine(rpd_output + "/FeatureLine_" + modelName + ".obj");
    outFeatureLine.precision(15);
    outFeatureLine.flags(ios::left | ios::fixed);
    outFeatureLine.fill('0');
    for (auto p : VersPC_ori) {
        outFeatureLine << "v " << p.transpose() << endl;
    }

    MyHalfEdgeModel FinalModel; // Final Model !!!
    std::cout << "Comput_RPD: Loading " << rpd_output +"/Remesh_" + modelName + ".obj" << " ..." << std::endl;
    FinalModel.ReadObjFile((rpd_output +"/Remesh_" + modelName + ".obj").c_str());

    auto Fedges = FinalModel.GetEdges();
    auto Ffaces = FinalModel.GetFaces();
    auto Fvecs = FinalModel.GetVertices();
    for (auto e : Fedges) {
        if (IsFeature[e.leftVert] && IsFeature[e.rightVert]) {
            auto f1 = e.indexOfFrontFace;
            auto f2 = Fedges[e.indexOfReverseEdge].indexOfFrontFace;
            Eigen::Vector3d Nf1, Nf2;

            //compute normal of f1
            auto v1 = Fvecs[Ffaces[f1].x()];
            auto v2 = Fvecs[Ffaces[f1].y()];
            auto v3 = Fvecs[Ffaces[f1].z()];
            Nf1 = (v2 - v1).cross(v3 - v1).normalized();

            //compute normal of f2
            v1 = Fvecs[Ffaces[f2].x()];
            v2 = Fvecs[Ffaces[f2].y()];
            v3 = Fvecs[Ffaces[f2].z()];
            Nf2 = (v2 - v1).cross(v3 - v1).normalized();

            //compute angle between f1 and f2
            double angle = acos(Nf1.dot(Nf2));

            // angle to drgee
            angle = angle * 180 / 3.1415926535;
            if (angle > 40 && angle < 140) {
                outFeatureLine << "l " << e.leftVert + 1 << " " << e.rightVert + 1 << "\n";
            }
        }
    }
    outFeatureLine.close();
    // output a single feature line model .
}
