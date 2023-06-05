#pragma once
#include <ctime>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <omp.h> 
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <vector>
#include "MyPointCloudModel.hpp"
#include "MyHalfEdgeModel.hpp"
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/IO/OBJ.h>
typedef CGAL::Simple_cartesian<double> K_T;
typedef K_T::Point_3 Point_T;

typedef CGAL::Polyhedron_3<K_T> Polyhedron;
typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> Primitive;
typedef CGAL::AABB_traits<K_T, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;

using namespace std;
using namespace nanoflann;


class DisjSet
{
private:
	std::vector<int> parent;
	std::vector<int> rank;

public:
	DisjSet(int max_size) : parent(std::vector<int>(max_size)),
		rank(std::vector<int>(max_size, 0))
	{
		for (int i = 0; i < max_size; ++i)
			parent[i] = i;
	}
	int find(int x)
	{
		return x == parent[x] ? x : (parent[x] = find(parent[x]));
	}
	void to_union(int x1, int x2)
	{
		int f1 = find(x1);
		int f2 = find(x2);
		if (rank[f1] > rank[f2])
			parent[f2] = f1;
		else
		{
			parent[f1] = f2;
			if (rank[f1] == rank[f2])
				++rank[f2];
		}
	}
	bool is_same(int e1, int e2)
	{
		return find(e1) == find(e2);
	}
};

void Comput_rnn(string modelpath, string modelname)
{
	string rnn_output = ".";

	vector<Eigen::Vector3d> VersPC_ori;
	vector<double> Weight;

	vector<vector<int>> neighboor;

	MyPointCloudModel PCmodel;
    std::cout << "Comput_rnn: Loading " << rnn_output + "/FinalPointCloud_"+ modelname + ".xyz" << " ..." << std::endl;
	PCmodel.ReadXYZFile((rnn_output + "/FinalPointCloud_"+ modelname + ".xyz").c_str(), true);

	int FeatureN = -1;
    std::cout << "Comput_rnn: Loading " << rnn_output + "/FeaturePointsNum_" + modelname + ".txt" << " ..." << std::endl;
	ifstream inFnum(rnn_output + "/FeaturePointsNum_" + modelname + ".txt");
	inFnum >> FeatureN;
	inFnum.close();

	double radis;
    std::cout << "Comput_rnn: Loading " << rnn_output + "/radis_" + modelname + ".txt" << " ..." << std::endl;
	ifstream inRnum(rnn_output + "/radis_" + modelname + ".txt");
	inRnum >> radis;
	inRnum.close();
	radis *= 4;
	//radis = 0.0025;

	MyHalfEdgeModel* PoissonModel = new MyHalfEdgeModel();
    std::cout << "Comput_rnn: Loading " << rnn_output + "/model_poisson_" + modelname + ".obj" << " ..." << std::endl;
	PoissonModel->ReadObjFile((rnn_output + "/model_poisson_" + modelname + ".obj").c_str());

	// Polyhedron polyhedron;
    // std::cout << "Comput_rnn: Loading " << rnn_output + "/model_poisson_" + modelname + ".off" << " ..." << std::endl;
	// std::ifstream input(rnn_output + "/model_poisson_" + modelname + ".off");

	VersPC_ori = PCmodel.GetVertices();

	// input >> polyhedron;
	// Tree tree(faces(polyhedron).first, faces(polyhedron).second, polyhedron);
	vector<Eigen::Vector3d> VersPC;
	for (int i = 0; i < VersPC_ori.size(); i++) {
		// Point_T query(VersPC_ori[i].x(), VersPC_ori[i].y(), VersPC_ori[i].z());
		// Point_T closest = tree.closest_point(query);
		// Eigen::Vector3d NewP(closest.x(), closest.y(), closest.z());
		VersPC.push_back(VersPC_ori[i]);
	}

	PointCloud<double> cloud;
	cloud.pts.resize(VersPC.size());
	for (int i = 0; i < VersPC.size(); i++) {
		cloud.pts[i].x = VersPC[i].x();
		cloud.pts[i].y = VersPC[i].y();
		cloud.pts[i].z = VersPC[i].z();
	}

	typedef KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<double, PointCloud<double> >,
		PointCloud<double>,
		3 /* dim */
	> my_kd_tree_t;

	my_kd_tree_t index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index.buildIndex();

	for (int i = 0; i < VersPC.size(); i++) {
		vector<int> tmp;
		neighboor.push_back(tmp);
		double query_pt[3] = { VersPC[i].x(), VersPC[i].y(), VersPC[i].z() };

		const double search_radius = static_cast<double>((radis) * (radis));
		std::vector<std::pair<uint32_t, double> > ret_matches;

		nanoflann::SearchParams params;
		const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches, params);

		for (size_t j = 0; j < nMatches; j++) {
			neighboor[i].push_back(ret_matches[j].first);
		}
	}

	DisjSet bcj(VersPC.size());

	for (int i = 0; i < FeatureN; i++) {
		for (auto pp : neighboor[i]) {
			double dis = (VersPC_ori[i] - VersPC_ori[pp]).norm();
			double dis1 = (VersPC[i] - VersPC[pp]).norm();
			if (dis <= 0.00015 || dis1 <= 0.00015 || VersPC[i] == VersPC[pp]) {
				if (!bcj.is_same(i, pp))
					bcj.to_union(i, pp);
			}
		}
	}
	vector<Eigen::Vector3d> MargedPoints_Ori;
	vector<Eigen::Vector3d> MargedPoints;
	vector<int> isFeature;

	map<int, vector<int>> cluster; // xxxx8888
	for (int i = 0; i < VersPC.size(); i++) {
		cluster[bcj.find(i)].push_back(i);
	}

	for (auto mp : cluster) {
		if (mp.second.size() > 0) {
			auto pts = mp.second;
			int tp = -1;
			double mndis = 9999999.0;
			for (int i = 0; i < pts.size(); i++) {
				double dis = 0;
				for (int j = 0; j < pts.size(); j++) {
					dis += (VersPC_ori[pts[i]] - VersPC_ori[pts[j]]).norm() + (VersPC[pts[i]] - VersPC[pts[j]]).norm();
				}
				if (dis < mndis) {
					mndis = dis;
					tp = pts[i];
				}
			}
			MargedPoints_Ori.push_back(VersPC_ori[tp]);
			MargedPoints.push_back(VersPC[tp]);

			isFeature.push_back(tp);
		}
	}
	auto Nors = PCmodel.GetNormals();

    std::cout << "Comput_rnn: Saving " << rnn_output + "/PoissonPoints_qc_" + modelname + ".xyz" << " ..." << std::endl;
	ofstream outNps(rnn_output + "/PoissonPoints_qc_" + modelname + ".xyz");
	outNps.precision(15);
	outNps.flags(ios::left | ios::fixed);
	outNps.fill('0');

    std::cout << "Comput_rnn: Saving " << rnn_output + "/OriPoints_qc_" + modelname + ".xyz" << " ..." << std::endl;
	ofstream outNpsori(rnn_output + "/OriPoints_qc_" + modelname + ".xyz");
	outNpsori.precision(15);
	outNpsori.flags(ios::left | ios::fixed);
	outNpsori.fill('0');

    std::cout << "Comput_rnn: Saving " << rnn_output + "/FeaturePointNum_" + modelname + ".txt" << " ..." << std::endl;
	ofstream outFnum(rnn_output + "/FeaturePointNum_" + modelname + ".txt");
	for (int ii = 0; ii < MargedPoints.size(); ii++) {
		outNps << MargedPoints[ii].transpose() << endl;
		outNpsori << MargedPoints_Ori[ii].transpose() << " " << Nors[isFeature[ii]].transpose() << endl;
		if (isFeature[ii] < FeatureN)
			outFnum << 1 << endl;
		else
			outFnum << 0 << endl;
	}
	outFnum.close();
	outNps.close();
	outNps.close();


	auto PoiVecs = PoissonModel->GetVertices();
	cloud.pts.resize(PoiVecs.size());

	for (int i = 0; i < PoiVecs.size(); i++) {
		cloud.pts[i].x = PoiVecs[i].x();
		cloud.pts[i].y = PoiVecs[i].y();
		cloud.pts[i].z = PoiVecs[i].z();
	}

	string filenPoisson = rnn_output + "/knn50_poisson_" + modelname + ".txt";
    std::cout << "Comput_rnn: Saving " << filenPoisson << " ..." << std::endl;
	ofstream outknnPoisson(filenPoisson);

	my_kd_tree_t index3(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index3.buildIndex();

	for (int i = 0; i < MargedPoints.size(); i++) {
		double query_pt[3] = { MargedPoints[i].x(), MargedPoints[i].y(), MargedPoints[i].z() };

		const double search_radius = static_cast<double>((radis * 1.4) * (radis * 1.4));
		std::vector<std::pair<uint32_t, double> > ret_matches;

		nanoflann::SearchParams params;
		size_t nMatches = index3.radiusSearch(&query_pt[0], search_radius, ret_matches, params);

		if (nMatches > 300)
			nMatches = size_t(300);
		outknnPoisson << nMatches << " ";

		for (size_t j = 0; j < nMatches; j++) {
			outknnPoisson << ret_matches[j].first << " ";
		}
		outknnPoisson << "\n";
	}
	outknnPoisson.close();
}
