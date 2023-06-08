#include <cstdio>
#include <cstdlib>
#include <vector>
//#include <chrono>

__global__ void calculateB(int nx, int ny, double dx, double dy, double dt, double rho, double nu, double* u, double* v, double* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;

    if (0 < i && i < nx-1 && 0 < j && j < ny-1 ) {
        b[j * nx + i] = rho * (1 / dt *
            ((u[j * nx + i + 1] - u[j * nx + i - 1]) / (2 * dx) + (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy)) -
            ((u[j * nx + i + 1] - u[j * nx + i - 1]) / (2 * dx)) * ((u[j * nx + i + 1] - u[j * nx + i - 1]) / (2 * dx))
            - 2 * ((u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2 * dy) *
                (v[j * nx + i + 1] - v[j * nx + i - 1]) / (2 * dx)) -
            ((v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy)) * ((v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy)));
    }
}

__global__ void calculateP(int nx, int ny, double dx, double dy, double* p, double* pn, double* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;

    if (0 < i && i < nx-1 && 0 < j && j < ny-1 ) {
        p[j * nx + i] = (dy * dy * (pn[j * nx + i + 1] + pn[j * nx + i - 1]) +
            dx * dx * (pn[(j + 1) * nx + i] + pn[(j - 1) * nx + i]) -
            b[j * nx + i] * dx * dx * dy * dy) / (2 * (dx * dx + dy * dy));
    }
}

__global__ void calculateUV(int nx, int ny, double dx, double dy, double dt, double rho, double nu, double* u, double* v, double* un, double* vn, double* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;

    if (0 < i && i < nx-1 && 0 < j && j < ny-1 ) {
        u[j * nx + i] = un[j * nx + i] - un[j * nx + i] * dt / dx * (un[j * nx + i] - un[j * nx + i - 1])
            - un[j * nx + i] * dt / dy * (un[j * nx + i] - un[(j - 1) * nx + i])
            - dt / (2 * rho * dx) * (p[j * nx + i + 1] - p[j * nx + i - 1])
            + nu * dt / (dx * dx) * (un[j * nx + i + 1] - 2 * un[j * nx + i] + un[j * nx + i - 1])
            + nu * dt / (dy * dy) * (un[(j + 1) * nx + i] - 2 * un[j * nx + i] + un[(j - 1) * nx + i]);

        v[j * nx + i] = vn[j * nx + i] - vn[j * nx + i] * dt / dx * (vn[j * nx + i] - vn[j * nx + i - 1])
            - vn[j * nx + i] * dt / dy * (vn[j * nx + i] - vn[(j - 1) * nx + i])
            - dt / (2 * rho * dx) * (p[j * nx + i + 1] - p[j * nx + i - 1])
            + nu * dt / (dx * dx) * (vn[j * nx + i + 1] - 2 * vn[j * nx + i] + vn[j * nx + i - 1])
            + nu * dt / (dy * dy) * (vn[(j + 1) * nx + i] - 2 * vn[j * nx + i] + vn[(j - 1) * nx + i]);
    }
}

void output(int step, int nx, int ny, double dx, double dy, double *u, double *v, double *p) {
    char filename[100];
    sprintf(filename, "result_%d.csv", step);

    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file %s\n", filename);
        return;
    }

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int index = j * nx + i;
            fprintf(file, "%f,%f,%f,%f,%f\n", i * dx, j * dy, u[index], v[index], p[index]);
        }
    }

    fclose(file);

    printf("Output file %s\n", filename);
}


int main() {
    const int nx = 41;
    const int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2. / (nx - 1);
    double dy = 2. / (ny - 1);
    double dt = 0.01;
    double rho = 1;
    double nu = 0.02;

    int size = nx * ny * sizeof(double);
    double* u, * v, * un, * vn, * b, * p, * pn;

    cudaMallocManaged(&u, size);
    cudaMallocManaged(&v, size);
    cudaMallocManaged(&un, size);
    cudaMallocManaged(&vn, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&p, size);
    cudaMallocManaged(&pn, size);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            u[j * nx + i] = 0;
            v[j * nx + i] = 0;
            un[j * nx + i] = 0;
            vn[j * nx + i] = 0;
            b[j * nx + i] = 0;
            p[j * nx + i] = 0;
            pn[j * nx + i] = 0;
        }
    }

    dim3 block( 1024, 1);
    dim3 grid( (nx + 1024 - 1) / 1024, ny);

    for (int n = 0; n < nt; n++) {
        //auto tic = chrono::steady_clock::now();

        calculateB<<<grid, block>>>(nx, ny, dx, dy, dt, rho, nu, u, v, b);
        cudaDeviceSynchronize();

        for (int it = 0; it < nit; it++) {
            cudaMemcpy(pn, p, size, cudaMemcpyDeviceToDevice);

            calculateP<<<grid, block>>>(nx, ny, dx, dy, p, pn, b);
            cudaDeviceSynchronize();

            for (int j = 1; j < ny - 1; j++) {
                p[j * nx + nx - 1] = p[j * nx + nx - 2];
                p[j * nx] = p[j * nx + 1];
            }

            for (int i = 1; i < nx - 1; i++) {
                p[i] = p[nx + i];
                p[(ny - 1) * nx + i] = p[(ny - 2) * nx + i];
            }
        }

        cudaMemcpy(un, u, size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(vn, v, size, cudaMemcpyDeviceToDevice);

        calculateUV<<<grid, block>>>(nx, ny, dx, dy, dt, rho, nu, u, v, un, vn, p);
        cudaDeviceSynchronize();

        for (int j = 1; j < ny - 1; j++) {
            u[j * nx] = 0;
            u[j * nx + nx - 1] = 0;
            v[j * nx] = 0;
            v[j * nx + nx - 1] = 0;
        }

        for (int i = 1; i < nx - 1; i++) {
            u[i] = 0;
            u[(ny - 1) * nx + i] = 1;
            v[i] = 0;
            v[(ny - 1) * nx + i] = 0;
        }

        //auto toc = chrono::steady_clock::now();
        //double time = chrono::duration<double>(toc - tic).count();
        //printf("step=%d: %lf s\n", n, time);

        if (n % 50 == 0){
            output(n, nx, ny, dx, dy, u, v, p);
        }
    }

    cudaFree(u);
    cudaFree(v);
    cudaFree(un);
    cudaFree(vn);
    cudaFree(b);
    cudaFree(p);
    cudaFree(pn);

    return 0;
}
