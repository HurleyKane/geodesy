#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

bool allclose(const MatrixXd& a, double atol) {
    bool mark = true;
    for (int i = 0; i < a.rows(); ++i) {
        for (int j = 0; j < a.cols(); ++j) {
            if (std::abs(a(i, j) - a(0, 0)) > atol) {
                mark = false;
                break;
            }
        }
        if (!mark) break;
    }
    return mark;
}

std::tuple<MatrixXd, std::vector<MatrixXd>, VectorXd, VectorXd> normal_equation(
    const std::vector<VectorXd>& L_list,
    const std::vector<MatrixXd>& B_list,
    const std::vector<MatrixXd>& W_list) {
    std::vector<MatrixXd> N_list(B_list.size());
    std::vector<MatrixXd> U_list(B_list.size());
    MatrixXd N = MatrixXd::Zero(B_list[0].cols(), B_list[0].cols());
    VectorXd U = VectorXd::Zero(B_list[0].cols());

    for (size_t i = 0; i < L_list.size(); ++i) {
        N_list[i] = B_list[i].transpose() * W_list[i] * B_list[i];
        U_list[i] = B_list[i].transpose() * W_list[i] * L_list[i];
        N += N_list[i];
        U += U_list[i];
    }

    VectorXd l;
    try {
        l = N.colPivHouseholderQr().solve(U);
    } catch (const std::exception& e) {
        return std::make_tuple(MatrixXd(), std::vector<MatrixXd>(), VectorXd(), VectorXd());
    }

    VectorXd delta(L_list.size());
    std::vector<VectorXd> v_list(L_list.size());
    for (size_t i = 0; i < L_list.size(); ++i) {
        v_list[i] = B_list[i] * l - L_list[i];
        delta(i) = v_list[i].transpose() * W_list[i] * v_list[i];
    }

    return std::make_tuple(N, N_list, delta, l);
}

std::tuple<VectorXd, MatrixXd, VectorXd> Helmert_one(
    const MatrixXd& N,
    const std::vector<MatrixXd>& N_list,
    const VectorXd& delta,
    const std::vector<int>& n_list) {
    MatrixXd gamma(n_list.size(), n_list.size());
    try {
        MatrixXd inv_N = N.inverse();
        for (size_t m = 0; m < N_list.size(); ++m) {
            for (size_t n = 0; n < N_list.size(); ++n) {
                if (m == n) {
                    gamma(m, n) = n_list[m] - 2 * (inv_N * N_list[m]).trace() +
                                  (inv_N * N_list[n] * inv_N * N_list[n]).trace();
                } else {
                    gamma(m, n) = (inv_N * N_list[m] * inv_N * N_list[n]).trace();
                }
            }
        }
        VectorXd sigma2 = gamma.inverse() * delta;
        return std::make_tuple(sigma2, gamma, delta);
    } catch (const std::exception& e) {
        return std::make_tuple(VectorXd(), MatrixXd(), VectorXd());
    }
}

std::tuple<VectorXd, int, std::vector<MatrixXd>> Helmert(
    const std::vector<VectorXd>& L_list,
    const std::vector<MatrixXd>& B_list,
    const std::vector<MatrixXd>& W_list,
    const std::vector<int>& n_list,
    double atol,
    int iterationNumThreshold) {
    MatrixXd N;
    std::vector<MatrixXd> N_list;
    VectorXd delta, l;
    std::tie(N, N_list, delta, l) = normal_equation(L_list, B_list, W_list);

    if (N.size() == 0) {
        return std::make_tuple(VectorXd(), 0, std::vector<MatrixXd>());
    }

    if (iterationNumThreshold == 0 && atol == 0) {
        return std::make_tuple(l, 0, W_list);
    }

    VectorXd sigma2;
    MatrixXd gamma;
    std::tie(sigma2, gamma, delta) = Helmert_one(N, N_list, delta, n_list);

    if (sigma2.size() == 0) {
        return std::make_tuple(VectorXd(), 0, std::vector<MatrixXd>());
    }

    int loop_num = 1;
    while (std::abs(sigma2.maxCoeff() - sigma2.minCoeff()) > atol) {
        int sub = sigma2.nonZeroIndices()[0];
        double unit_sigma2 = sigma2(sub);
        std::vector<MatrixXd> new_W_list(W_list.size());
        for (size_t i = 0; i < L_list.size(); ++i) {
            if (sigma2(i) != 0) {
                new_W_list[i] = unit_sigma2 / sigma2(i) * W_list[i];
            } else {
                new_W_list[i] = W_list[i];
            }
        }

        std::tie(N, N_list, delta, l) = normal_equation(L_list, B_list, new_W_list);
        if (N.size() == 0) {
            return std::make_tuple(VectorXd(), 0, std::vector<MatrixXd>());
        }

        std::tie(sigma2, gamma, delta) = Helmert_one(N, N_list, delta, n_list);
        if (sigma2.size() == 0) {
            return std::make_tuple(VectorXd(), 0, std::vector<MatrixXd>());
        }

        loop_num++;
        if (loop_num > iterationNumThreshold) {
            return std::make_tuple(l, loop_num, new_W_list);
        }
    }

    return std::make_tuple(l, loop_num, new_W_list);
}

int main() {
    // 测角系数矩阵B1 12*4
    MatrixXd B1(12, 4);
    B1 << 0.5532, -0.8100, 0, 0,
          0.2434, 0.5528, 0, 0,
          -0.7966, 0.2572, 0, 0,
          -0.2434, -0.5528, 0, 0,
          0.6298, 0.6368, 0, 0,
          -0.3864, -0.0840, 0, 0,
          0.7966, -0.2572, -0.2244, -0.3379,
          -0.8350, -0.1523, 0.0384, 0.4095,
          0.0384, 0.4095, 0.1860, -0.0716,
          -0.0384, -0.4095, 0.2998, 0.1901,
          -0.3480, 0.3255, -0.0384, -0.4095,
          0.3864, 0.0840, -0.2614, 0.2194;

    // 测边系数矩阵B2 6*4
    MatrixXd B2(6, 4);
    B2 << 0.3072, 0.9516, 0, 0,
          -0.9152, 0.4030, 0, 0,
          0.2124, -0.9722, 0, 0,
          0, 0, -0.6429, -0.7660,
          0, 0, -0.8330, 0.5532,
          0.9956, -0.0934, -0.9956, 0.0934;

    std::vector<MatrixXd> B = {B1, B2};

    // 测角常数项L1 12*1
    VectorXd L1(12);
    L1 << 0.18, -0.53, 3.15, 0.23, -2.44, 1.01, 2.68, -4.58, 2.80, -3.10, 8.04, -1.14;

    // 测边常数项L2 6*1
    VectorXd L2(6);
    L2 << -0.84, 1.54, -3.93, 2.15, -12.58, -8.21;

    std::vector<VectorXd> L = {L1, L2};

    // 测角和测边个数
    int n1 = 12;
    int n2 = 6;
    std::vector<int> n = {n1, n2};

    // 初始定权，测角中误差=1.5，测边中误差=2.0
    double sita1 = 1.5 * 1.5;
    double sita2 = 2.0 * 2.0;
    MatrixXd P1 = MatrixXd::Identity(12, 12);
    MatrixXd P2 = sita1 / sita2 * MatrixXd::Identity(6, 6);
    std::vector<MatrixXd> P = {P1, P2};

    // Helmert方差分量估计算法
    VectorXd l;
    int loop_num;
    std::vector<MatrixXd> W_list;
    std::tie(l, loop_num, W_list) = Helmert(L, B, P, n, 0.01, 30);

    std::cout << "迭代" << loop_num << "次\n未知参数平差值：\n" << l << std::endl;

    return 0;
}
