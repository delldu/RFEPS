#include <iostream>
#include <fstream>
#include <functional>
#include <random>
#include <omp.h>
#include <vector>
#include <BGAL/Optimization/LinearSystem/LinearSystem.h>
#include <BGAL/Optimization/ALGLIB/optimization.h>
#include <BGAL/Optimization/LBFGS/LBFGS.h>
#include <BGAL/BaseShape/Point.h>
#include <BGAL/BaseShape/Polygon.h>
#include <BGAL/Integral/Integral.h>
#include <BGAL/Model/ManifoldModel.h>
#include <BGAL/Model/Model_Iterator.h>
#include <BGAL/Optimization/GradientDescent/GradientDescent.h>
#include <BGAL/BaseShape/KDTree.h>
#include "nanoflann.hpp"
#include "nanoflann/examples/utils.h"
#include "BGAL/Optimization/ALGLIB/dataanalysis.h"
#include "MyRPD.hpp"
#include "MyRPD_rnn.hpp"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/poisson_surface_reconstruction.h>
#include <CGAL/IO/read_points.h>

using namespace nanoflann;
using namespace alglib;

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point1;
typedef Kernel::Vector_3 Vector;
typedef std::pair<Point1, Vector> Pwn;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron1;

int rnnnum = 80;

void Poisson(string modelpath, string model)
{
	clock_t start, end;
	start = clock();

	string poisson_output = ".";

	std::vector<Pwn> points;

    std::cout << "Poisson: Loading " << poisson_output + "/Denoise_Final_" + model+".xyz" << " ..." << std::endl;
	if (!CGAL::IO::read_points(CGAL::data_file_path(poisson_output + "/Denoise_Final_" + model + ".xyz"), std::back_inserter(points),
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>())
		.normal_map(CGAL::Second_of_pair_property_map<Pwn>())))
	{
		std::cerr << "Error: cannot read input file!" << std::endl;
		return;
	}

    std::cout << "Poisson: Saving to " << poisson_output + "/model_poisson_" + model + ".off" << " ..." << std::endl;
	Polyhedron1 output_mesh;
	double average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>
		(points, 6, CGAL::parameters::point_map(CGAL::First_of_pair_property_map<Pwn>()));
	if (CGAL::poisson_surface_reconstruction_delaunay
	(points.begin(), points.end(),
		CGAL::First_of_pair_property_map<Pwn>(),
		CGAL::Second_of_pair_property_map<Pwn>(),
		output_mesh, average_spacing))
	{
		std::ofstream out(poisson_output + "/model_poisson_" + model + ".off");
		out << output_mesh;
        out.close();
	}

    std::cout << "Poisson: Loading " << poisson_output + "/model_poisson_" + model + ".off" << " ..." << std::endl;
	ifstream in(modelpath + "/model_poisson_" + model + ".off");
	vector<Eigen::Vector3d> pts;
	vector<Eigen::Vector3i> facs;
	
	string line;
	in >> line;
	int q, w, e;
	in >> q >> w >> e;
	for (int i = 0; i < q; i++) {
		double x, y, z;
		in >> x >> y >> z;
		pts.push_back(Eigen::Vector3d(x, y, z));
	}
	for (int i = 0; i < w; i++) {
		int a, x, y, z;
		in >>a>> x >> y >> z;
		facs.push_back(Eigen::Vector3i(x, y, z));
	}

    std::cout << "Poisson: Saving to " << poisson_output + "/model_poisson_" + model + ".obj" << " ..." << std::endl;
	std::ofstream outobj(poisson_output + "/model_poisson_" + model + ".obj");
	
	for (int i = 0; i < q; i++) {
		outobj << "v " << pts[i].transpose() << endl;
	}
	for (int i = 0; i < w; i++) {
		outobj << "f " << (facs[i]+ Eigen::Vector3i(1,1,1)).transpose() << endl;
	}
	outobj.close();

	end = clock();
	double endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "Poisson Running Time: " << endtime << endl;
}

// (sin(𝑢) cos(𝑣), sin(𝑢) sin(𝑣), cos(𝑢))
pair<double, double>  V3toV2(Eigen::Vector3d nor)
{
	double u = 0.0, v = 0.0;
	u = acos(nor.z());
	if (u == 0) {
		v = 0;
	} else {
		double tmp1 = abs(acos(nor.x() / sin(u)));
		double tmp2 = abs(asin(nor.y() / sin(u)));
		if (isnan(tmp1)) {
			if (isnan(tmp2)) {
				v = 0.0;
				pair<double, double> p(u, v);
				return p;
			} else {
				v = tmp2;
				pair<double, double> p(u, v);
				return p;
			}
		}
		if (isnan(tmp2)) {
			if (isnan(tmp1)) {
				v = 0.0;
				pair<double, double> p(u, v);
				return p;
			} else {
				v = tmp1;
				pair<double, double> p(u, v);
				return p;
			}
		}
		Eigen::Vector3d n11(sin(u) * cos(tmp1), sin(u) * sin(tmp1), cos(u));
		Eigen::Vector3d n22(sin(u) * cos(tmp2), sin(u) * sin(tmp2), cos(u));
		double tot1, tot2;
		tot1 = (n11 - nor).x() * (n11 - nor).x() + (n11 - nor).y() * (n11 - nor).y() + (n11 - nor).z() * (n11 - nor).z();
		tot2 = (n22 - nor).x() * (n22 - nor).x() + (n22 - nor).y() * (n22 - nor).y() + (n22 - nor).z() * (n22 - nor).z();
		if (tot1 < tot2) {
			v = tmp1;
		} else {
			v = tmp2;
		}
		if (abs(sin(u) * cos(v) - nor.x()) > 0.1) {
			u = -1.0 * u;
		}
		if (abs(sin(u) * sin(v) - nor.y()) > 0.1) {
			v = -1.0 * v;
		}
	}
	pair<double, double> p(u, v);
	return p;
}


void Test(string modelpath, string model, bool ifdenoise)
{
	clock_t start, end;
	vector<Eigen::Vector3d> Vall, Nall;
	vector<vector<int>> neighboor;

	ofstream out;
	string test_output = "."; 

    // model -- "01_82-block"
	ifstream in;
	if (ifdenoise) {
        std::cout << "Test: Loading " << test_output + "/Denoise_" + model + ".xyz" << " ..." << std::endl;
		in.open(test_output + "/Denoise_" + model + ".xyz");
	} else {
        std::cout << "Test: Loading " << modelpath + "/" + model + ".xyz" << " ..." << std::endl;
		in.open(modelpath + "/" + model + ".xyz");
	}

	while (!in.eof()) {
		Eigen::Vector3d p, n;
		in >> p.x() >> p.y() >> p.z() >> n.x() >> n.y() >> n.z();
		Vall.push_back(p);
		Nall.push_back(n);
	}

	int r = Vall.size();
	double radius = 0.002;
	double lambda = 0.05;

	PointCloud<double> cloud;
	int maxk = -9999999;
	cloud.pts.resize(r);
	for (int i = 0; i < r; i++) {
		cloud.pts[i].x = Vall[i].x();
		cloud.pts[i].y = Vall[i].y();
		cloud.pts[i].z = Vall[i].z();
	}
	typedef KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<double, PointCloud<double> >,
		PointCloud<double>,
		3 /* dim */
	> my_kd_tree_t;

	my_kd_tree_t index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index.buildIndex();

	do {
		radius += 0.001;
		neighboor.clear();
		for (int i = 0; i < r; i++) {
			vector<int> tmp;
			neighboor.push_back(tmp);
			double query_pt[3] = { Vall[i].x(), Vall[i].y(), Vall[i].z() };
			const double search_radius = static_cast<double>((radius) * (radius));
			std::vector<std::pair<uint32_t, double> > ret_matches;
			nanoflann::SearchParams params;
			const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches, params);
			for (size_t j = 0; j < nMatches; j++) {
				neighboor[i].push_back(ret_matches[j].first);
			}
		}
		maxk = -99999;
		auto neighboor_denoise = neighboor;
		for (int i = 0; i < r; i++) {
			int k = neighboor[i].size();
			neighboor_denoise[i].clear();
			for (int j = 0; j < k; j++) {
				int pid = neighboor[i][j];
				double dis = (Nall[i] - Nall[pid]).norm();
				//if (dis < 1.5)
				{
					neighboor_denoise[i].push_back(pid);
				}
			}
			maxk = max(maxk, int(neighboor_denoise[i].size()));
		}
		neighboor = neighboor_denoise;
		// cout << "maxk: " << maxk << endl;
	}  while (maxk < rnnnum );

	// add par later

	start = clock();
	map<int, double> R3;
	for (int iter = 0; iter < r; iter++)
		R3[iter] = 0;

	omp_set_num_threads(24);
#pragma omp parallel for schedule(dynamic, 20)  //part 1
	for (int iter = 0; iter < r; iter++) // part 1
	{
		//cout << iter << endl;
		std::function<void(const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr)>
            fop_lambda = [&](const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr) -> void
		{
			int k = neighboor[iter].size();
			int r = Vall.size();
			vector<double > w;
			for (int i = 0; i < k; i++)
				w.push_back(0);
#if 1 // xxxx8888
			for (int i = 0; i < k; i++) {
				double dis = (Vall[iter] - Vall[neighboor[iter][i]]).norm();
				if (dis < 0.0001) {
					w[i] = 0.0;
				} else {
					w[i] = 1.0 / (dis * dis);
				}
			}
#else            
			for (int i = 0; i < k; i++) {
				w[i] = 1.0;
			}
#endif
			double u1 = x[k], v1 = x[k + 1], u2 = x[k + 2], v2 = x[k + 3];
			Eigen::Vector3d n1(sin(u1) * cos(v1), sin(u1) * sin(v1), cos(u1));
			Eigen::Vector3d n2(sin(u2) * cos(v2), sin(u2) * sin(v2), cos(u2));
			n1.normalize();
			n2.normalize();
			func = 0;
			for (int i = 0; i < k; i++) {
				auto Nj = Nall[neighboor[iter][i]];
				func += x[i] * w[i] * (pow((Nj.x() - sin(u1) * cos(v1)), 2) + pow(Nj.y() - sin(u1) * sin(v1), 2) + pow(Nj.z() - cos(u1), 2))
					+ (1.0 - x[i]) * w[i] * (pow((Nj.x() - sin(u2) * cos(v2)), 2) + pow(Nj.y() - sin(u2) * sin(v2), 2) + pow(Nj.z() - cos(u2), 2));
			}

			for (int i = 0; i < k; i++) {
				//double a = abs(cos(x[k]))* abs(cos(x[k]));
				auto Nj = Nall[neighboor[iter][i]];
				grad[i] = w[i] * (pow((Nj.x() - sin(u1) * cos(v1)), 2) + pow(Nj.y() - sin(u1) * sin(v1), 2) + pow(Nj.z() - cos(u1), 2))
					- w[i] * (pow((Nj.x() - sin(u2) * cos(v2)), 2) + pow(Nj.y() - sin(u2) * sin(v2), 2) + pow(Nj.z() - cos(u2), 2));
			}

			grad[k] = 0; grad[k + 1] = 0; grad[k + 2] = 0; grad[k + 3] = 0;

			for (int i = 0; i < k; i++) {
				auto Nj = Nall[neighboor[iter][i]];
				grad[k] += 2 * x[i] * sin(u1) * w[i] * (Nj.z() - cos(u1))
					- (2 * x[i] * cos(v1) * cos(u1) * w[i] * (Nj.x() - sin(u1) * cos(v1))
						+ 2 * x[i] * cos(u1) * w[i] * sin(v1) * (Nj.y() - sin(u1) * sin(v1)));

				grad[k + 1] += 2 * x[i] * sin(u1) * sin(v1) * w[i] * (Nj.x() - sin(u1) * cos(v1))
					- 2 * x[i] * sin(u1) * cos(v1) * w[i] * (Nj.y() - sin(u1) * sin(v1));

				grad[k + 2] += 2 * w[i] * sin(u2) * (1 - x[i]) * (Nj.z() - cos(u2))
					- (2 * w[i] * cos(u2) * cos(v2) * (1 - x[i]) * (Nj.x() - sin(u2) * cos(v2))
						+ 2 * w[i] * cos(u2) * sin(v2) * (1 - x[i]) * (Nj.y() - sin(u2) * sin(v2)));

				grad[k + 3] += 2 * sin(u2) * sin(v2) * (1 - x[i]) * w[i] * (Nj.x() - sin(u2) * cos(v2))
					- 2 * sin(u2) * cos(v2) * (1 - x[i]) * w[i] * (Nj.y() - sin(u2) * sin(v2));
			}
		}; // fop_lambda

		int k = neighboor[iter].size();
		if (k < 5)
			continue;
		// kmeans

		clusterizerstate s;
		kmeansreport rep;
		//real_2d_array xy = "[[1,1],[1,2],[4,1],[2,3],[4,1.5]]";
		real_2d_array xy;
		xy.setlength(k, 3);
		for (int i = 0; i < k; i++) {
			xy[i][0] = Nall[neighboor[iter][i]].x();
			xy[i][1] = Nall[neighboor[iter][i]].y();
			xy[i][2] = Nall[neighboor[iter][i]].z();
		}

		alglib::clusterizercreate(s);
		alglib::clusterizersetpoints(s, xy, 2);
		alglib::clusterizersetkmeanslimits(s, 10, 0);
		alglib::clusterizerrunkmeans(s, 2, rep);// ?
		Eigen::Vector3d C1, C2;
		if (int(rep.terminationtype) == -3) {
			C1 = Nall[neighboor[iter][0]];
			C2 = Nall[neighboor[iter][0]];
		} else {
			C1.x() = rep.c[0][0];
			C1.y() = rep.c[0][1];
			C1.z() = rep.c[0][2];
			C2.x() = rep.c[1][0];
			C2.y() = rep.c[1][1];
			C2.z() = rep.c[1][2];
		}

		real_1d_array x0;
		x0.setlength(k + 4);
		real_1d_array s0;
		s0.setlength(k + 4);
		for (int i = 0; i < k; i++) {
			x0[i] = 0.5;
			s0[i] = 1;
		}
		for (int i = k; i < k + 4; i++) {
			x0[i] = 0;
			s0[i] = 1;
		}

		C2.normalize();
		C1.normalize();

		auto Q = V3toV2(C1);
		x0[k] = Q.first;
		x0[k + 1] = Q.second;

		Q = V3toV2(C2);
		x0[k + 2] = Q.first;
		x0[k + 3] = Q.second;
		//kmeans[iter] = make_pair(make_pair(x0[k], x0[k + 1]), make_pair(x0[k + 2], x0[k + 3]));

		real_1d_array bndl;
		real_1d_array bndu;
		bndl.setlength(k + 4);
		bndu.setlength(k + 4);

		real_2d_array c;
		c.setlength(1, neighboor[iter].size() + 5);
		integer_1d_array ct = "[0]";
		for (int i = 0; i < k; i++) {
			bndl[i] = 0;
			bndu[i] = 1;
			c[0][i] = 1;
		}
		for (int i = k; i < k + 4; i++) {
			bndl[i] = -99999999999;
			bndu[i] = 99999999999;
			c[0][i] = 0;
		}
		c[0][k + 4] = double(k) / 2.0;
		minbleicstate state;
		double epsg = 0;
		double epsf = 0;
		double epsx = 0;
		ae_int_t maxits = 0.000001;
		alglib::minbleiccreate(x0, state);
		alglib::minbleicsetlc(state, c, ct);
		alglib::minbleicsetbc(state, bndl, bndu);
		alglib::minbleicsetscale(state, s0);
		alglib::minbleicsetcond(state, epsg, epsf, epsx, maxits);

		alglib::minbleicoptguardsmoothness(state);
		alglib::minbleicoptguardgradient(state, 0.0001);

		minbleicreport rep2;
		if (1) {
			alglib::minbleicoptimize(state, fop_lambda);
			alglib::minbleicresults(state, x0, rep2);
			double mn = 0;
			real_1d_array G_tmp;
			G_tmp.setlength(k + 4);
			fop_lambda(x0, mn, G_tmp, nullptr);
		}
		

		double u1 = x0[k], v1 = x0[k + 1], u2 = x0[k + 2], v2 = x0[k + 3];
		Eigen::Vector3d n1(sin(u1) * cos(v1), sin(u1) * sin(v1), cos(u1));
		Eigen::Vector3d n2(sin(u2) * cos(v2), sin(u2) * sin(v2), cos(u2));

		double dis = sqrt((n1.x() - n2.x()) * (n1.x() - n2.x()) + (n1.y() - n2.y()) * (n1.y() - n2.y()) + (n1.z() - n2.z()) * (n1.z() - n2.z()));

		//compute angle between n1 n2
		double angle = acos(n1.dot(n2));
		// angel to drgee
		angle = angle * 180 / 3.1415926535;

		if (dis < 30 / 100.0) // importent
			R3[iter] = 1; // normal point
	} // Compute R3 Done !!!

	end = clock();
	double endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "T2 Running Time: " << endtime << endl;
	cout << "T2 Running Time: " << endtime * 1000 << " ms " << endl;
	

	// part2 normal
	//KNNSC
	auto neighboor_M = neighboor;
	cout << "ASDA\n";
	for (int i = 0; i < r; i++) {
		int k = neighboor[i].size();
		if (k < 5)
			continue;

		if (R3[i] == 1)
            continue;


		int pid = 1;
		k = neighboor_M[i].size();
		neighboor_M[i].clear();

		double query_pt[3] = { Vall[i].x(), Vall[i].y(), Vall[i].z() };
		const double search_radius = static_cast<double>((radius * 3) * (radius * 3));
		std::vector<std::pair<uint32_t, double> > ret_matches;
		nanoflann::SearchParams params;
		const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches, params);

		neighboor_M[i].push_back(i);
		for (size_t j = 0; j < nMatches; j++) {
			if (R3[ret_matches[j].first] == 1) {
				neighboor_M[i].push_back(ret_matches[j].first);
				pid++;
				if (pid == k)
					break;
			}
		}
		double tmpr = radius * 3;
		while (pid != k) {
			tmpr *= 4;
			pid = 1;
			neighboor_M[i].clear();
			const double search_radius = static_cast<double>((tmpr) * (tmpr));
			std::vector<std::pair<uint32_t, double> > ret_matches;
			nanoflann::SearchParams params;
			const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches, params);

			neighboor_M[i].push_back(i);
			for (size_t j = 0; j < nMatches; j++) {
				if (R3[ret_matches[j].first] == 1) {
					neighboor_M[i].push_back(ret_matches[j].first);
					pid++;
					if (pid == k)
						break;
				}
			}
		}
	} // Update neighboor_M ?
	cout << "ASDA\n";

	map<int, Eigen::Vector3d> Nall_new;
	for (int iter = 0; iter < r; iter++) {
		Nall_new[iter] = Nall[iter];
	}
	
	start = clock();
#pragma omp parallel for schedule(dynamic, 20)  //part 2
	for (int iter = 0; iter < r; iter++) {
		int k = neighboor[iter].size();
		if (k < 5)
			continue;

		k = neighboor_M[iter].size();
		vector<int> v1, cnt;
		for (int i = 0; i < k; i++) {
			v1.push_back(i);
			cnt.push_back(0);
		}
		for (int i = 0; i < k - 1; i++) {
			for (int j = i + 1; j < k; j++) {
				double dij = (Nall[neighboor_M[iter][i]] - Nall[neighboor_M[iter][j]]).norm();
				if (dij < lambda)
					v1[j] = v1[i];
			}
		}
		for (int i = 0; i < k; i++) {
			cnt[v1[i]] = cnt[v1[i]] + 1;
		}
		int mx = -99999;
		int center = 0;
		for (int i = 0; i < k; i++) {
			if (cnt[i] > mx) {
				mx = cnt[i];
				center = i;
			}
		}
		Eigen::Vector3d nor(0, 0, 0);
		for (int i = 0; i < k; i++) {
			if (v1[i] == center)
				nor = nor + Nall[neighboor_M[iter][i]];
		}
		nor.x() /= double(mx);
		nor.y() /= double(mx);
		nor.z() /= double(mx);
		nor.normalize();

		Nall_new[iter] = nor;
	} // Update Nall_new



	end = clock();
	endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "T3 Running Time: " << endtime << endl;
	cout << "T3 Running Time: " << endtime * 1000 << " ms " << endl;


	
	auto neighboor_denoise = neighboor;
	std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)>
        fg = [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
	{
		double func = 0.0;

		for (int i = 0; i < r; i++)
			g(i) = 0;

		for (int i = 0; i < r; i++) {
			int k = neighboor_denoise[i].size();
			if (k < 5)
				continue;

			for (int j = 0; j < k; j++) {
				int pj = neighboor_denoise[i][j];
				Eigen::Vector3d Pj = Vall[pj] + X(pj) * Nall_new[pj];
				Eigen::Vector3d Pi = Vall[i] + X(i) * Nall_new[i];
				Eigen::Vector3d Vij = Pj - Pi;

				Eigen::Matrix3d a;
				a = Vij * Vij.transpose();
				Eigen::Vector3d ans = a * Nall_new[i];
				func += ans.norm();

				auto t0 = Vij;

				double t1 = t0.transpose() * Nall_new[i];
				double t2 = Nall_new[i].transpose() * t0;
				double t3 = t0.transpose() * t0;
				double t4 = Nall_new[i].transpose() * Nall_new[i];


				g[i] += -1.0 * (2.0 * t1 * t1 * t2 + 2.0 * t1 * t3 * t4);
				double t5 = t0.transpose() * Nall_new[pj];
				double t6 = Nall_new[i].transpose() * Nall_new[pj];

				g[pj] += 2.0 * t1 * t5 * t2 + 2.0 * t1 * t3 * t6;
			}
		}
		return func;
	}; // fg

	for (int i = 0; i < r; i++) {
		int k = neighboor[i].size();
		if (k < 5)
			continue;

		neighboor_denoise[i].clear();
		for (int j = 0; j < k; j++) {
			int pid = neighboor[i][j];
			double dis = (Nall_new[i] - Nall_new[pid]).norm();
			if (dis < 0.2) {
				neighboor_denoise[i].push_back(pid);
			}
		}
	}
	
	vector<Eigen::Vector3d> Vall_new = Vall;
	start = clock();
	if(1) {
		BGAL::_LBFGS::_Parameter param = BGAL::_LBFGS::_Parameter();
		param.epsilon = 1e-4;
		param.is_show = true;
		param.max_iteration = 35;
		param.max_linearsearch = 5;
		BGAL::_LBFGS lbfgs(param);

		Eigen::VectorXd iterX(r);
		for (int i = 0; i < r; i++) {
			iterX(i) = 0;
		}
		int n = lbfgs.minimize(fg, iterX); // Optimization
		std::cout << "Optimization Minimize n: " << n << std::endl;

		for (int i = 0; i < r; i++) {
			Vall_new[i] = Vall[i] + iterX(i) * Nall_new[i];
		}
	} // Update Vall_new ?
	
	end = clock();
	endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "T4 Running Time: " << endtime << endl;
	cout << "T4 Running Time: " << endtime * 1000 << " ms " << endl;

    std::cout << "Test: Saving " << test_output +"/Denoise_Final_" + model + ".xyz" << " ..." << std::endl;
	ofstream fout(test_output +"/Denoise_Final_" + model + ".xyz");
	for (size_t i = 0; i < r; i++) {
		int k = neighboor_denoise[i].size();
		if (k < 5)
			continue;
	
		fout << Vall_new[i].transpose() << " " << Nall_new[i].transpose() << endl;
	}
    fout.close();

	Vall = Vall_new;

	// part 3
	map<int, bool> flag3;
	for (int iter = 0; iter < r; iter++) {
		flag3[iter] = (R3[iter] == 1)? 1 : 0;
	}
	cout << "Part3 pre compute\n";
	for (int iter = 0; iter < r; iter++) {
		if (flag3[iter])
			continue;
		int k = neighboor[iter].size();
		if (k < 5)
			continue;
		
		double maxAng = -9999.0;
		for (int i = 1; i < k; i++) {
			double sigma = acos(Nall_new[neighboor[iter][i]].dot(Nall_new[neighboor[iter][0]]) / (Nall_new[neighboor[iter][i]].norm() * Nall_new[neighboor[iter][0]].norm()));
			sigma = sigma / EIGEN_PI * 180.0;
			maxAng = max(maxAng, sigma);
		}

		if (maxAng < 60) {
			flag3[iter] = 1;
			continue;
		}

	}


	bool iw = 0;
	start = clock();
	map<int, Eigen::Vector3d> NewPoints;
//#pragma omp parallel for schedule(dynamic, 20) //part 3
	cout << "Begin Part3 feature points...\n";
	for (int iter = 0; iter < r; iter++)
	{
		if (flag3[iter])
			continue;

		int k = neighboor[iter].size();
		if (k < 5)
			continue;
	
		real_1d_array x0;
		x0.setlength(3);
		x0[0] = Vall[iter].x();
		x0[1] = Vall[iter].y();
		x0[2] = Vall[iter].z();
		real_1d_array s0 = "[1,1,1]";

		std::function<void(const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr)>
            fop_z_lambda = [&](const alglib::real_1d_array& x, double& func, alglib::real_1d_array& grad, void* ptr) -> void
		{
			int k = neighboor[iter].size();
			int r = Vall.size();

			Eigen::Vector3d z(x[0], x[1], x[2]);
			double mu = 0.01;
			func = 0;
			for (int i = 0; i < k; i++) {
				auto zp = Vall[neighboor[iter][i]] - z;
				func += pow(zp.dot(Nall_new[neighboor[iter][i]]), 2);
				//func += x[i] * w[i] * (Nall[neighboor[iter][i]] - n1).squaredNorm()  + (1 - x[i]) * w[i] * (Nall[neighboor[iter][i]] - n2).squaredNorm();
			}
			func = func + mu * (Vall[iter] - z).squaredNorm();

			Eigen::Vector3d g(0, 0, 0);
			g = 2 * mu * (z - Vall[iter]);
			for (int i = 0; i < k; i++) {
				//double a = abs(cos(x[k]))* abs(cos(x[k]));
				auto Pj = Vall[neighboor[iter][i]];
				auto nj = Nall_new[neighboor[iter][i]];
				g = g + 2 * (z - Pj).dot(nj) * nj;
			}

			grad[0] = g.x();
			grad[1] = g.y();
			grad[2] = g.z();
		}; // fop_z_lambda

		minbleicstate state;
		double epsg = 0;
		double epsf = 0;
		double epsx = 0;
		ae_int_t maxits = 0;
		alglib::minbleiccreate(x0, state);
		alglib::minbleicsetscale(state, s0);
		alglib::minbleicsetcond(state, epsg, epsf, epsx, maxits);

		alglib::minbleicoptguardsmoothness(state);
		alglib::minbleicoptguardgradient(state, 0.000001);

		minbleicreport rep2;
		alglib::minbleicoptimize(state, fop_z_lambda);
		alglib::minbleicresults(state, x0, rep2);
		double mn = 0;
		real_1d_array G_tmp;
		G_tmp.setlength(3);
		fop_z_lambda(x0, mn, G_tmp, nullptr);
		double dis = (Vall[iter] - Eigen::Vector3d(x0[0], x0[1], x0[2])).norm();
		if (dis > 0.0001) {
			NewPoints[iter] = Eigen::Vector3d(x0[0], x0[1], x0[2]);
		} else {
			flag3[iter] = true;
		}
		//NewPoints[iter] = Eigen::Vector3d(x0[0], x0[1], x0[2]);
	}

	for (auto np : NewPoints) {
		if (flag3[np.first])
			continue;
		double dis = (Vall[np.first] - np.second).norm();
		if (dis > radius*2 || dis < 0.00001)
			flag3[np.first] = 1;
	}

    std::cout << "Test: Saving " << test_output + "/FinalPointCloud1_" + model + ".xyz" << " ..." << std::endl;
	out.open(test_output + "/FinalPointCloud1_" + model + ".xyz");
	vector<bool> flag2(r, 0);
	for (auto np : NewPoints) {
		if (flag3[np.first])
			continue;
		out << np.second.transpose() <<" "<< Nall_new[np.first].transpose() << endl;
		flag2[np.first] = 1;
	}
	for (size_t i = 0; i < r; i++) {
		int k = neighboor_denoise[i].size();
		if (k < 5)
			continue;
		
		if(!flag2[i])
			out << Vall[i].transpose() << " " << Nall_new[i].transpose() << endl;
	}

	out.close();

    std::cout << "Test: Saving " << test_output + "/FeaturePointsNum_" + model + ".txt" << " ..." << std::endl;
	out.open(test_output + "/FeaturePointsNum_" + model + ".txt");
	int cnt = 0;
	for (auto np : NewPoints) {
		if (flag3[np.first])
			continue;
		cnt++;
	}
	out << cnt << endl;
	out.close();

    std::cout << "Test: Saving " << test_output + "/radis_" + model + ".txt" << " ..." << std::endl;
	out.open(test_output + "/radis_" + model + ".txt");
    out << radius << endl;
    out.close();

	end = clock();
	endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "T5 Running Time: " << endtime << endl;
	cout << "T5 Running Time: " << endtime * 1000 << " ms " << endl;
}


void Denoise(string modelpath, string model)
{
	clock_t start, end;

	vector<Eigen::Vector3d> Vall, Nall;
	vector<vector<int>> neighboor;

	ofstream out;
	
	string denoise_output = ".";

    std::cout << "Denoise: Loading " << modelpath + "/" + model + ".xyz" << " ..." << std::endl;
	ifstream in(modelpath + "/" + model + ".xyz");
	while (!in.eof()) {
		Eigen::Vector3d p, n;
		in >> p.x() >> p.y() >> p.z() >> n.x() >> n.y() >> n.z();
		{
			Vall.push_back(p);
			Nall.push_back(n);
		}
	}

	int r = Vall.size();
	for (int i = 0; i < Vall.size(); i++) {
		Nall[i].normalize();
	}

	double radius = 0.001;
	double lambda = 0.05;

	// rnn
	PointCloud<double> cloud;
	int maxk = -9999999;

	cloud.pts.resize(Vall.size());
	for (int i = 0; i < Vall.size(); i++) {
		cloud.pts[i].x = Vall[i].x();
		cloud.pts[i].y = Vall[i].y();
		cloud.pts[i].z = Vall[i].z();
	}
	typedef KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<double, PointCloud<double> >,
		PointCloud<double>,
		3 /* dim */
	> my_kd_tree_t;

	my_kd_tree_t   index(3 /*dim*/, cloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
	index.buildIndex();

    int search_times = 0;
	do {
        // std::cout << "Radius Search ... " << search_times + 1 << std::endl;
		radius += 0.001;
		neighboor.clear();
		maxk = -99999;
		for (int i = 0; i < Vall.size(); i++) {
			vector<int> tmp;
			neighboor.push_back(tmp);
			double query_pt[3] = { Vall[i].x(), Vall[i].y(), Vall[i].z() };
			const double search_radius = static_cast<double>((radius) * (radius));
			std::vector<std::pair<uint32_t, double> > ret_matches;
			nanoflann::SearchParams params;
			const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches, params);
			for (size_t j = 0; j < nMatches; j++) {
				neighboor[i].push_back(ret_matches[j].first);
			}
			maxk = max(maxk, int(nMatches));
		}
        search_times++;
	}  while (maxk < rnnnum);


	start = clock();
	auto Vall_new = Vall;
	auto Nall_new = Nall;

	std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)>
        fg = [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
	{
		double func = 0.0;
		double lambda = 0.01;

		for (int i = 0; i < r*3; i++)
			g(i) = 0;

		auto Nall_tmp = Nall_new;
		for (int i = 0; i < r; i++) {
			double u = X(r + i * 2);
			double v = X(r + i * 2 + 1);
			Nall_tmp[i].x() = sin(u) * cos(v);
			Nall_tmp[i].y() = sin(u) * sin(v);
			Nall_tmp[i].z() = cos(u);
		}
		
		for (int i = 0; i < r; i++) {
			int k = neighboor[i].size();
			if (k < 5)
				continue;

			for (int j = 0; j < k; j++) {
				int pj = neighboor[i][j];
				Eigen::Vector3d Pj = Vall[pj] + X(pj) * Nall_tmp[pj];
				Eigen::Vector3d Pi = Vall[i] + X(i) * Nall_tmp[i];
				Eigen::Vector3d Vij = Pj - Pi;

				Eigen::Matrix3d a;
				a = Vij * Vij.transpose();
				Eigen::Vector3d ans = a * Nall_tmp[i];
				func += ans.norm();

				auto t0 = Vij;

				double t1 = t0.transpose() * Nall_tmp[i];
				double t2 = Nall_tmp[i].transpose() * t0;
				double t3 = t0.transpose() * t0;
				double t4 = Nall_tmp[i].transpose() * Nall_tmp[i];

				g[i] += -1.0 * (2.0 * t1 * t1 * t2 + 2.0 * t1 * t3 * t4);
				double t5 = t0.transpose() * Nall_tmp[pj];
				double t6 = Nall_tmp[i].transpose() * Nall_tmp[pj];

				g[pj] += 2.0 * t1 * t5 * t2 + 2.0 * t1 * t3 * t6;

				// g of ni
				{
					Eigen::Vector3d t0 = Vij;
					Eigen::Vector3d t1 = t0.transpose() * Nall_tmp[i] * t0;
					double t2 = 2 * X(i);
					double t3 = t0.transpose() * t0;
					double t4 = Nall_tmp[i].transpose() * t0;
					Eigen::Vector3d gni(0, 0, 0);
					
					gni = 2.0 * t3 * t1 -(t2 * t4 * t1 + t2 * t3 * t4 * Nall_tmp[i]);
					
					g[r + i * 2] += gni.x() * cos(X(r + i * 2)) * cos(X(r + i * 2 + 1))  + gni.y() * cos(X(r + i * 2)) * sin(X(r + i * 2 + 1)) + -1.0 * gni.z() * sin(X(r + i * 2));
					g[r + i * 2 + 1] += -1.0* gni.x() * sin(X(r + i * 2)) * sin(X(r + i * 2 + 1)) + gni.y() * sin(X(r + i * 2)) * cos(X(r + i * 2 + 1));
					
				}
				// g of nj
				{
					Eigen::Vector3d t0 = Vij;
					double t1 = 2*X(pj);
					double t2 = Nall_tmp[i].transpose() * t0;
					double t3 = t0.transpose() * Nall_tmp[i];
					double t4 = t0.transpose() * t0;
					Eigen::Vector3d gnj(0,0,0);
					gnj = t1 * t2 * t3 * t0 + t1 * t2 * t4*Nall_tmp[i];
					g[r + pj * 2] += gnj.x() * cos(X(r + pj * 2)) * cos(X(r + pj * 2 + 1)) + gnj.y() * cos(X(r + pj * 2)) * sin(X(r + pj * 2 + 1)) + -1.0 * gnj.z() * sin(X(r + pj * 2));
					g[r + pj * 2 + 1] += -1.0 * gnj.x() * sin(X(r + pj * 2)) * sin(X(r + pj * 2 + 1)) + gnj.y() * sin(X(r + pj * 2)) * cos(X(r + pj * 2 + 1));
				}
			}
		}
		for (int i = 0; i < r; i++) {
			Eigen::Vector3d Pi = Vall[i] + X(i) * Nall_tmp[i];
			func += lambda* (Pi - Vall[i]).squaredNorm();
			double tt = (Pi - Vall[i]).transpose() * Nall_tmp[i];
			g[i] += lambda * 2 * tt;
		}

		return func;
	};

	
	BGAL::_LBFGS::_Parameter param = BGAL::_LBFGS::_Parameter();
	param.epsilon = 1e-5;
	param.is_show = true;
	BGAL::_LBFGS lbfgs(param);

	Eigen::VectorXd iterX(r*3);
	for (int i = 0; i < r; i++) {
		iterX(i) = 0;
	}

	for (int i = 0; i < r; i++) {
		auto Q = V3toV2(Nall_new[i]);
		iterX(r+i*2) = Q.first;
		iterX(r+i*2+1) = Q.second;
	}
	int n = lbfgs.minimize(fg, iterX);
	std::cout << "Optimization Minimize n: " << n << std::endl;

	for (int i = 0; i < r; i++) {
		double u = iterX(r+i*2);
		double v = iterX(r+i*2+1);
		Nall_new[i].x() = sin(u) * cos(v);
		Nall_new[i].y() = sin(u) * sin(v);
		Nall_new[i].z() = cos(u);
		Vall_new[i] = Vall[i] + iterX(i) * Nall_new[i];
	}
	end = clock();

	double endtime = (double)(end - start) / CLOCKS_PER_SEC;
	cout << "Denoise Running Time: " << endtime << endl;
	cout << "Denoise Running Time: " << endtime * 1000 << " ms " << endl;

    std::cout << "Denoise: Saving " << denoise_output + "/Denoise_" + model +".xyz" << " ..." << std::endl;
	ofstream fout(denoise_output + "/Denoise_" + model +".xyz");
	for (size_t i = 0; i < r; i++) {
		int k = neighboor[i].size();
		if (k < 5)
			continue;
		fout << Vall_new[i].transpose() << " " << Nall_new[i].transpose() << endl;
	}
	fout.close();
}

int main()
{
	string modelpath = "../../data";
	string modelname = "01_82-block";

	rnnnum = 60;
	bool noise = 1;
	if (noise) {
		rnnnum *= 2.0;
		Denoise(modelpath, modelname);
	}

	Test(modelpath, modelname, noise);
	Poisson(modelpath, modelname);
	Comput_rnn(modelpath, modelname);
	Comput_RPD(modelpath, modelname);
	
	return 0;
}
