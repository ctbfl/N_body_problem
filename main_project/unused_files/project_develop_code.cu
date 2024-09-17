#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <chrono>
#include <thread>


/*
  设备信息
  Device Number: 0
  Device name: NVIDIA GeForce RTX 4090
  Total global memory: 25756696576 bytes
  Shared memory per block: 49152 bytes
  Registers per block: 65536
  Warp size: 32
  Memory pitch: 2147483647 bytes
  Max threads per block: 1024
  Max threads dimensions: x = 1024, y = 1024, z = 64
  Max grid dimensions: x = 2147483647, y = 65535, z = 65535
  Clock rate: 2550000 kHz
  Total constant memory: 65536 bytes
  Compute capability: 8.9
  Texture alignment: 512 bytes
  Device overlap: 1
  Multi-processor count: 128
  Kernel execution timeout enabled: 1
  Integrated GPU sharing host memory: 0
  Can map host memory with cudaHostAlloc/cudaHostGetDevicePointer: 1
  Compute mode: 0
  Concurrent kernels: 1
  ECC enabled: 0
  PCI bus ID: 1
  PCI device ID: 0
  PCI domain ID: 0
  TCC driver mode: 0
  Async engine count: 1
  Unified addressing: 1
  Memory clock rate: 10501000 kHz
  Memory bus width: 384 bits
  L2 cache size: 75497472 bytes
  Max threads per multi-processor: 1536
  Stream priorities supported: 1
  Global L1 cache supported: 1
  Local L1 cache supported: 1
  Shared memory per multi-processor: 102400 bytes
  Registers per multi-processor: 65536
  Managed memory: 1
  Is multi-GPU board: 0
  Multi-GPU board group ID: 0
 */


#define G 1
#define TIME_TICK 0.008
#define SLOW_DOWN_FACTOR 0.001
#define BLOCK_SIZE 256 // 块的边长
#define EPSILON 1e-6
#define CHECK_IDX 20256
#define Y_LOW_BOUND 8192
#define Y_HIGH_BOUND 8447
#define detect_X 10000
#define detect_Y 5000
#define ADDITIONAL_ID 512
#define VERSION 5
#define SPEED_MAX 10
#define FLOAT_COMPENSATE 10
#define MAX_BODIES 100000
#define DEFAULT_DATASET 0
// 2代表朴素方法(4ms)，
// 3代表单线程法(非常多ms)，4代表第一次改进后的自定义方法(9ms)，5代表第二次改进后的自定义方法（最终版本!）(2ms)
// 0代表最初的版本（30ms, 有bug），1代表最初版本的双精度版本（60ms，有bug），
typedef float real;


float fov = 45.0f; // 视场角

bool isMousePressed = false;
double lastMouseX, lastMouseY;

float cameraDistance = 1.0;
glm::vec2 sphereCoords(0.0f, 0.0f); // 球坐标：theta和phi
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 1.0f);
glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float  scaleFactors[3];

bool m_initialized;

// 速度buffer
real* m_deviceVelocity;

unsigned int m_numBodies;
unsigned int real_body_nums;

// 双缓冲处理，后来发现没必要删去了，因为渲染非常非常快
unsigned int m_pbo[2];
unsigned int m_currentRead;
unsigned int m_currentWrite;
cudaGraphicsResource* m_pGRes[2]; // 与GL的缓冲区相绑定的




struct Header
{
    double time;
    int nbodies;
    int ndimension;
    int nsph;
    int ndark;
    int nstar;
};

struct DarkParticle
{
    real mass;
    real pos[3];
    real vel[3];
    real eps;
    int phi;
};

struct StarParticle
{
    real mass;
    real pos[3];
    real vel[3];
    real metals;
    real tform;
    real eps;
    int phi;
};

void initialize(int numBodies)
{
    m_numBodies = numBodies;
    unsigned int memSize = m_numBodies * 4 * sizeof(real);
    std::vector<real> position(m_numBodies * 4, 0);

    // generate double buffers.
    glGenBuffers(2, m_pbo);

    // initilize devices' buffers.
    glBindBuffer(GL_ARRAY_BUFFER, m_pbo[0]);
    glBufferData(GL_ARRAY_BUFFER, memSize, &position[0], GL_DYNAMIC_DRAW); // 初始化整个缓冲区
    int size = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size);
    if ((unsigned)size != memSize)
        fprintf(stderr, "WARNING: Pixel Buffer Object allocation failed!\n");
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ARRAY_BUFFER, m_pbo[1]);
    glBufferData(GL_ARRAY_BUFFER, memSize, &position[0], GL_DYNAMIC_DRAW); // 初始化整个缓冲区
    size = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size);
    if ((unsigned)size != memSize)
        fprintf(stderr, "WARNING: Pixel Buffer Object allocation failed!\n");
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // allocate velocity buffer.
    cudaMalloc((void**)&m_deviceVelocity, memSize); // 为全体速度数据速度分配空间
    cudaGraphicsGLRegisterBuffer(&m_pGRes[0], m_pbo[0], cudaGraphicsMapFlagsNone); // 将数字m_pbo[0]与缓冲区m_pGRes[0]绑定
    cudaGraphicsGLRegisterBuffer(&m_pGRes[1], m_pbo[1], cudaGraphicsMapFlagsNone); // 将数字m_pbo[1]与缓冲区m_pGRes[1]绑定

    m_initialized = true;
}

void setParticlesPosition(real* data)
{
    // 初始化所有粒子的位置
    if (!m_initialized)
        return;
    m_currentRead = 0;
    m_currentWrite = 1;
    glBindBuffer(GL_ARRAY_BUFFER, m_pbo[m_currentRead]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 4 * sizeof(real) * m_numBodies, data); // 更新缓冲区的内容，从offset=0位置开始，更新数据量为所有天体的位置
    int size = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size); // 获取缓冲区大小
    if ((unsigned)size != 4 * (sizeof(real) * m_numBodies))
        printf( "WARNING: Pixel Buffer Object download failed!\n");
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void setParticlesVelocity(real* data)
{
    // 初始化粒子速度
    if (!m_initialized)
        return;
    m_currentRead = 0;
    m_currentWrite = 1;

    cudaMemcpy(m_deviceVelocity, data, m_numBodies * 4 * sizeof(real),
        cudaMemcpyHostToDevice); //  m_deviceVelocity为当前的索引指针
}

void readTipsyFile(const std::string& fileName, std::vector<real>& positions, std::vector<real>& velocities, std::vector<int>& bodiesIds, int& nTotal, int& nFirst, int& nSecond, int& nThird)
{
    char fullFileName[256];
    sprintf(fullFileName, "%s", fileName.c_str());

    std::ifstream input(fullFileName, std::ios::in | std::ios::binary);
    if (!input.is_open())
    {
        std::cout << "Couldn't not open the tipsy file: " << fileName << std::endl;
        return;
    }

    Header header;
    input.read((char*)&header, sizeof(header));

    int idummy;
    glm::fvec4 pos;
    glm::fvec4 vel;

    nTotal = header.nbodies;
    real_body_nums = nTotal;
    nFirst = header.ndark;
    nSecond = header.nstar;
    nThird = header.nsph;
    positions.reserve(4 * nTotal);
    velocities.reserve(4 * nTotal);
    DarkParticle dark;
    StarParticle star;
    for (int index = 0; index < nTotal; ++index)
    {
        if (index < nFirst)
        {
            // dark particle.
            input.read((char*)&dark, sizeof(dark));
            vel.w = dark.eps;
            pos.w = dark.mass;
            pos.x = dark.pos[0];
            pos.y = dark.pos[1];
            pos.z = dark.pos[2];
            vel.x = dark.vel[0];
            vel.y = dark.vel[1];
            vel.z = dark.vel[2];
            idummy = dark.phi;
        }
        else
        {
            // star particle.
            input.read((char*)&star, sizeof(star));
            vel.w = star.eps;
            pos.w = star.mass;
            pos.x = star.pos[0];
            pos.y = star.pos[1];
            pos.z = star.pos[2];
            vel.x = star.vel[0];
            vel.y = star.vel[1];
            vel.z = star.vel[2];
            idummy = star.phi;
        }
        if (isnan(pos.x) || isnan(pos.y) || isnan(pos.z))
            std::cout << "Nan Error\n";
        positions.push_back(pos.x);
        positions.push_back(pos.y);
        positions.push_back(pos.z);
        positions.push_back(pos.w);
        velocities.push_back(vel.x);
        velocities.push_back(vel.y);
        velocities.push_back(vel.z);
        velocities.push_back(vel.w);
        bodiesIds.push_back(idummy);
    }

    // 填充到 BLOCK_SIZE 的倍数+1
    int newTotal = nTotal;
    if (nTotal % BLOCK_SIZE)
        newTotal = (nTotal / BLOCK_SIZE + 1) * BLOCK_SIZE;
    newTotal = newTotal + 1;//（由于实现的原因，需要总数量是BLOCK_SIZE + 1个）
    for (int index = nTotal; index < newTotal; ++index)
    {
        positions.push_back(0.0f);
        positions.push_back(0.0f);
        positions.push_back(0.0f);
        positions.push_back(0.0f);
        velocities.push_back(0.0f);
        velocities.push_back(0.0f);
        velocities.push_back(0.0f);
        velocities.push_back(0.0f);
        bodiesIds.push_back(index);
        ++nFirst;
    }
    nTotal = newTotal;


    input.close();
}

// 从文件读入天体数据，并加载到GPU上以及buffer上
// 该函数结束的时候，所有天体的位置数据都在m_pGRes[0]里面，初始速度则在m_deviceVelocity这个临时存储区里面。
void loadTipsyFile(const std::string& path)
{

    std::vector<int> ids;
    std::vector<real> positions;
    std::vector<real> velocities;

    int nBodies = 0;
    int nFirst = 0, nSecond = 0, nThird = 0;

    readTipsyFile(path, positions, velocities, ids,
        nBodies, nFirst, nSecond, nThird); 
    // 运行完毕后，positions中按照xyz weight的方式依次存入位置数据， velocities中按照xyz eps的方式依次存入速度数据，ids存入势能数据。
    // nFirst是dark的数量,nSecond是star的数量,nThird是SPH粒子的数量
    printf("nFirst=%d, nSecond=%d, nThird=%d\n", nFirst, nSecond, nThird);
    initialize(nBodies); // 按照数量初始化buffer
    setParticlesPosition(&positions[0]); // 传入的是全体数据的开头位置的指针, 将全体位置&重量数据保存到一个缓冲区里
    setParticlesVelocity(&velocities[0]); //传入的是全体数据的开头位置的指针，将全体速度数据写入到内存空间里
}

void readTabFile(const std::string& fileName, std::vector<real>& positions, std::vector<real>& velocities, int& nTotal){
    std::ifstream input;
    input.open(fileName, std::ifstream::in);
    if (input.fail())
    {
        std::cout << "Couldn't not open the tipsy file: " << fileName << std::endl;
        return;
    }
    std::string line;
    glm::vec4 pos, vel;
    nTotal = 0;
    bool first = true;
    bool second = true;
    while (!input.eof())
    {
        getline(input, line);
        std::istringstream iss(line.c_str());
        // mass x y z vx vy vz.
        iss >> pos.w >> pos.x >> pos.y >> pos.z >> vel.x >> vel.y >> vel.z;
        positions.push_back(pos.x);
        positions.push_back(pos.y);
        positions.push_back(pos.z);
        positions.push_back(pos.w);
        velocities.push_back(vel.x);
        velocities.push_back(vel.y);
        velocities.push_back(vel.z);
        velocities.push_back(vel.w);
        ++nTotal;
    }
    real_body_nums = nTotal;
    // round up to a multiple of 256 bodies.
    int newTotal = nTotal;
    if (nTotal % BLOCK_SIZE)
        newTotal = (nTotal / BLOCK_SIZE + 1) * BLOCK_SIZE;
    newTotal = newTotal + 1;
    for (int index = nTotal; index < newTotal; ++index)
    {
        positions.push_back(0.0f);
        positions.push_back(0.0f);
        positions.push_back(0.0f);
        positions.push_back(0.0f);
        velocities.push_back(0.0f);
        velocities.push_back(0.0f);
        velocities.push_back(0.0f);
        velocities.push_back(0.0f);
    }
    nTotal = newTotal;

    input.close();
}

void loadTabFile(const std::string& path){
    std::vector<real> positions;
    std::vector<real> velocities;

    int nBodies = 0;
    readTabFile(path, positions, velocities, nBodies);

    initialize(nBodies);
    setParticlesPosition(&positions[0]);
    setParticlesVelocity(&velocities[0]);
}

void readDatFile(const std::string& fileName, std::vector<real>& positions, std::vector<real>& velocities, int& nTotal)
{
    std::ifstream input;
    input.open(fileName, std::ifstream::in);
    if (input.fail())
    {
        std::cout << "Couldn't not open the tipsy file: " << fileName << std::endl;
        return;
    }
    std::string line;
    glm::vec4 pos, vel;
    nTotal = 0;
    pos.w = 1.0f;
    while (!input.eof())
    {
        getline(input, line);
        if (line.empty())
            continue;
        std::istringstream iss(line.c_str());
        // z y x vz vy vz.
        iss >> pos.z >> pos.y >> pos.x >> vel.z >> vel.y >> vel.x;
        positions.push_back(pos.x);
        positions.push_back(pos.y);
        positions.push_back(pos.z);
        positions.push_back(pos.w);
        velocities.push_back(vel.x);
        velocities.push_back(vel.y);
        velocities.push_back(vel.z);
        velocities.push_back(vel.w);
        ++nTotal;
    }
    real_body_nums = nTotal;
    // round up to a multiple of BLOCK_SIZE bodies.
    int newTotal = nTotal;
    if (nTotal % BLOCK_SIZE)
        newTotal = (nTotal / BLOCK_SIZE + 1) * BLOCK_SIZE;
    newTotal++;
    for (int index = nTotal; index < newTotal; ++index)
    {
        positions.push_back(0.0f);
        positions.push_back(0.0f);
        positions.push_back(0.0f);
        positions.push_back(0.0f);
        velocities.push_back(0.0f);
        velocities.push_back(0.0f);
        velocities.push_back(0.0f);
        velocities.push_back(0.0f);
    }
    nTotal = newTotal;

    input.close();
}

void loadDatFile(const std::string& path){
    // load the bodies from a dat file.
    std::vector<real> positions;
    std::vector<real> velocities;

    int nBodies = 0;
    readDatFile(path, positions, velocities, nBodies);

    initialize(nBodies);
    setParticlesPosition(&positions[0]);
    setParticlesVelocity(&velocities[0]);
}

void readSnapFile( const std::string& fileName, std::vector<real>& positions, std::vector<real>& velocities, int& nTotal)
{
    std::ifstream input;
    input.open(fileName, std::ifstream::in);
    if (input.fail())
    {
        std::cout << "Couldn't not open the tipsy file: " << fileName << std::endl;
        return;
    }
    std::string line;
    glm::vec4 pos, vel;
    nTotal = 0;

    // nbodies.
    getline(input, line);
    std::istringstream iss(line.c_str());
    iss >> nTotal;
    positions.resize(4 * nTotal);
    velocities.resize(4 * nTotal);

    // ndim.
    int dim = 0;
    getline(input, line);
    iss.str(line.c_str());
    iss >> dim;

    // time.
    float time;
    getline(input, line);
    iss.str(line.c_str());
    iss >> time;

    //masses.
    for (int x = 0; x < nTotal; ++x)
    {
        if (input.eof())
        {
            std::cout << "It's not a normal snap file:" << fileName << std::endl;
            return;
        }
        getline(input, line);
        std::istringstream iss(line.c_str());
        iss >> pos.w;
        positions[4 * x + 3] = pos.w;
    }

    // pos.
    for (int x = 0; x < nTotal; ++x)
    {
        if (input.eof())
        {
            std::cout << "It's not a normal snap file:" << fileName << std::endl;
            return;
        }
        getline(input, line);
        std::istringstream iss(line.c_str());
        iss >> pos.x >> pos.y >> pos.z;
        positions[4 * x + 0] = pos.x;
        positions[4 * x + 1] = pos.y;
        positions[4 * x + 2] = pos.z;
    }

    // vel.
    for (int x = 0; x < nTotal; ++x)
    {
        if (input.eof())
        {
            std::cout << "It's not a normal snap file:" << fileName << std::endl;
            return;
        }
        getline(input, line);
        std::istringstream iss(line.c_str());
        iss >> vel.x >> vel.y >> vel.z;
        velocities[4 * x + 0] = vel.x;
        velocities[4 * x + 1] = vel.y;
        velocities[4 * x + 2] = vel.z;
    }

    // eps.
    for (int x = 0; x < nTotal; ++x)
    {
        if (input.eof())
        {
            std::cout << "It's not a normal snap file:" << fileName << std::endl;
            return;
        }
        getline(input, line);
        std::istringstream iss(line.c_str());
        iss >> vel.w;
        velocities[4 * x + 3] = vel.w;
    }

    // round up to a multiple of BLOCK_SIZE bodies.
    real_body_nums = nTotal;
    int newTotal = nTotal;
    if (nTotal % BLOCK_SIZE)
        newTotal = (nTotal / BLOCK_SIZE + 1) * BLOCK_SIZE;
    newTotal++;
    for (int index = nTotal; index < newTotal; ++index)
    {
        positions.push_back(0.0f);
        positions.push_back(0.0f);
        positions.push_back(0.0f);
        positions.push_back(0.0f);
        velocities.push_back(0.0f);
        velocities.push_back(0.0f);
        velocities.push_back(0.0f);
        velocities.push_back(0.0f);
    }
    nTotal = newTotal;

    input.close();
}

void loadSnapFile(const std::string& path)
{
    std::vector<real> positions;
    std::vector<real> velocities;

    int nBodies = 0;
    readSnapFile(path, positions, velocities, nBodies);

    initialize(nBodies);
    setParticlesPosition(&positions[0]);
    setParticlesVelocity(&velocities[0]);
}

// 读取文件内容到字符串
std::string readFile(const char* filePath) {
    std::string content;
    std::ifstream fileStream(filePath, std::ios::in);

    if (!fileStream.is_open()) {
        std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
        return "";
    }

    std::string line = "";
    while (!fileStream.eof()) {
        std::getline(fileStream, line);
        content.append(line + "\n");
    }

    fileStream.close();
    return content;
}

// 检查着色器和程序的编译/链接错误
void checkShaderError(GLuint shader, GLuint flag, bool isProgram, const std::string& errorMessage) {
    GLint success = 0;
    GLchar error[1024] = { 0 };

    if (isProgram) {
        glGetProgramiv(shader, flag, &success);
    }
    else {
        glGetShaderiv(shader, flag, &success);
    }

    if (success == GL_FALSE) {
        if (isProgram)
            glGetProgramInfoLog(shader, sizeof(error), NULL, error);
        else
            glGetShaderInfoLog(shader, sizeof(error), NULL, error);

        std::cerr << errorMessage << ": '" << error << "'" << std::endl;
    }
}

// 加载并编译着色器，链接到程序中
GLuint loadShader(const char* vertex_path, const char* fragment_path) {
    std::string vertShaderStr = readFile(vertex_path);
    std::string fragShaderStr = readFile(fragment_path);
    const char* vertShaderSrc = vertShaderStr.c_str();
    const char* fragShaderSrc = fragShaderStr.c_str();

    GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(vertShader, 1, &vertShaderSrc, NULL);
    glShaderSource(fragShader, 1, &fragShaderSrc, NULL);

    glCompileShader(vertShader);
    checkShaderError(vertShader, GL_COMPILE_STATUS, false, "Vertex shader compilation failed");

    glCompileShader(fragShader);
    checkShaderError(fragShader, GL_COMPILE_STATUS, false, "Fragment shader compilation failed");

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);
    glLinkProgram(shaderProgram);
    checkShaderError(shaderProgram, GL_LINK_STATUS, true, "Shader program linking failed");

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    return shaderProgram;
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}


/*  #region -------------------方法A--------------------  */

// 计算出i和j之间的引力在引力数组中的index
__device__ int calculate_index(int N, int i, int j) {
    if (i > j) { // 交换i和j，使得j>=i
        int t = i;
        i = j;
        j = t;
    } 
    return i * (2 * N - i - 1) / 2 + j - i - 1;
}

// 使用block_id反推算group_id_x,group_id_y
__device__ void calculate_group_x_y(int hori_group_nums, int block_id, int* group_id_x, int* group_id_y) {
    int layer = 0;
    int line_len = hori_group_nums;
    int last = block_id + 1;  // 将block_id从0开始调整为从1开始

    while (last > 0) {
        last -= line_len;
        if (last > 0) {
            layer++;
            line_len--;
        }
    }
    last = last + line_len;
    *group_id_x = layer;
    *group_id_y = layer + last - 1;
}

// 从块间坐标(group_x, group_y)，转化为全矩阵坐标(x,y)
__device__ int calculate_matrix_index(int group_x, int group_y, int* x, int* y, int block_size) {
    *x = block_size * group_x;
    *y = block_size * group_y + 1;
}

// 计算单个引力
// x: 物体a的xyzw信息指针
// y: 物体b的xyzw信息指针
__device__ float3 cal_single_gravity(real* x, real* y) {
    // 数据组织：x,y,z,w（每个数据一个real类型大小）
    // F = G*m1*m2/r^2
    real dx = x[0] - y[0];
    real dy = x[1] - y[1];
    real dz = x[2] - y[2];
    real distance_squared = dx * dx + dy * dy + dz * dz;

    // 防止距离为零的情况，并避免除以零
    if (distance_squared < EPSILON) {
        distance_squared = EPSILON;
    }
    real distance = sqrt(distance_squared);
    real force =  G * x[3] * y[3] / (distance* distance* distance);
    float3 force_vector;
    force_vector.x = force * dx;
    force_vector.y = force * dy;
    force_vector.z = force * dz;
    return force_vector;
}

// 计算引力主函数
// 调用方式
// cal_gravity<<<block_num , block_edge_len, size>>>(gravity_array, (real*)position_and_weight, block_edge_len, hori_group_nums);
// size: 共享内存的大小，应为2*block_edge_len*4*sizeof(real)
// hori_group_nums: 一行中的最多总任务块数量
// block_edge_len: 任务块的边长
// gravity_array: 全局内存中的，引力数组，总长度为m(m-1)/2*size(real),计算结果要保存在这里
// position_and_weight: 全局内存中的位置质量数组,每个点占4*size(real)大小，总长度为m*4*size(real)
__global__ void cal_gravity(float3* gravity_array, real* position_and_weight, int block_edge_len, int hori_group_nums, int N) {
    extern __shared__ real mass[];  // 声明一个动态共享内存，保存该任务块要用到的质量&位置数组 （只保存各个列对应的物体，行对应的物体直接保存在线程内部）
    // size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程应处理的天体索引
    int look_at = 16;
    // if (idx == look_at)printf("noew thread 0\n");
    // 任务块号是blockIdx.x
    // 线程在块内的编号是threadIdx.x
    // 块内线程总数为block_edge_len (= blockDim.x)
    // 根据任务块号和块内编号，加载数据到共享内存
    int group_id_x, group_id_y; // 块在块矩阵中的坐标
    int point_x, point_y, point_y_start; //块的左上角点在任务矩阵中的坐标
    calculate_group_x_y(hori_group_nums,blockIdx.x,&group_id_x, &group_id_y);
    //printf("idx = %d, block_id = %d, group_id_x = %d, group_id_y = %d\n", idx, blockIdx.x,group_id_x, group_id_y);
    calculate_matrix_index(group_id_x, group_id_y, &point_x, &point_y, block_edge_len);// 先找到在总matrix中的起始点，以定位当前处理的行
    //if (blockIdx.x % look_at == 0)printf("idx = %d, block_id = %d, group_id_x=%d, group_id_y=%d, point.x = %d, point.y = %d\n",idx, blockIdx.x, group_id_x, group_id_y, point_x, point_y);
    point_x = point_x + threadIdx.x; // 左上角坐标x->真实物体坐标x，同时也是本次计算的行号
    point_y_start = point_y;         // 本次计算的列起始号
    point_y = point_y + threadIdx.x; // 左上角坐标x->真实物体坐标y
    //// 未处理边缘区块数量不足的情况
    //if (point_x >= 20481) {
    //    printf("idx = %d,block_id = %d, group_id_x = %d, group_id_y = %d, x=%d, y=%d out of range\n", idx, blockIdx.x, group_id_x, group_id_y, point_y, point_y);
    //}
    //if (point_y >= 20481) {
    //    printf("idx = %d,block_id = %d, group_id_x = %d, group_id_y = %d, x=%d, y=%d out of range\n", idx, blockIdx.x, group_id_x, group_id_y, point_x, point_y);
    //}
    int share_pointer = threadIdx.x << 2;  // 等同于 threadIdx.x * 4
    int body_pointer_x = point_x << 2;     // 等同于 point_x * 4
    int body_pointer_y = point_y << 2;     // 等同于 point_y * 4
    real body_data[4] = {};     // 线程独享的数据
    for (int i = 0;i < 4;i++) { // 每个线程加载到共享内存一个点，block_edge_len个线程一共加载block_edge_len个点的数据
        body_data[i] = position_and_weight[body_pointer_x + i];
        mass[share_pointer + i] = position_and_weight[body_pointer_y + i];  
    }
    
    // 加载结束后,组内同步
    __syncthreads();
    
    // 正式开始重力计算
    // 计算任务是（point_x， point_y_start）一直到 （point_x， point_y_start+block_edge_len-1）
    // 也就是线程计算的是一行
    // 行对应的物体数据存在变量body_data里面，列对应的数据直接依次存在mass里面
    real* mass_pointer = mass;
    for (int i = 0;i < block_edge_len;i++) { // 边算边存，最大利用存储带宽
        float3 result = cal_single_gravity(body_data, mass_pointer);
        //if (i==0&&idx % 10000 == 0) {
        //    printf("idx = %zu, result = (%.4f, %.4f, %.4f)\n", result.x, result.y, result.z);
        //}
        int gravity_array_index = calculate_index(N, point_x, point_y_start + i);
        gravity_array[gravity_array_index] = result;
        mass_pointer += 4;
    }
}

// 将引力相加
// gravity_matrix_array，各个天体两两之间的引力矩阵，被拉平成一个数组，要结合calculate_index调用。
//  gravity_matrix_array[calculate_index(N, i, j)]就是引力矩阵中的第(i,j)位置的元素
// gravity_sum_array，结果目标数组，长度为N，存储其他天体对该天体的引力总和
// float3有三个分量 x,y,z，均为real类型
// 暂时是没有做进一步的并行。
__global__ void add_up_gravity(float3* gravity_matrix_array, float3* gravity_sum_array, int N) {
    // 由于计算index的时候i,j会自动处理，所以不用特别在意。但是不能使用(i,i)的值，因为对应的index是无效的。
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程应处理的天体索引
    
    if (i < N) {
        float3 sum = { 0.0, 0.0, 0.0 };  // 初始化引力和为0

        for (int j = 0; j < N; j++) {
            if (i != j) {  // 避免计算天体对自身的引力
                int index = calculate_index(N, i, j);  // 获取拉平后数组中的索引
                float3 force = gravity_matrix_array[index];  // 从数组中取出引力

                sum.x += force.x;  // 累加x方向的引力
                sum.y += force.y;  // 累加y方向的引力
                sum.z += force.z;  // 累加z方向的引力
            }
        }

        gravity_sum_array[i] = sum;  // 将计算出的引力和存储到结果数组中
        //if (i % 10000 == 0) {
        //    //printf("i=%d force.x=%.4f\n", i, force.x);
        //    printf("i=%d gravity_sum  sum=（%.4f，%.4f ，%.4f ）\n", i, sum.x, sum.y, sum.z);
        //}
        //if (i == N - 2)printf("i=%d gravity_sum  sum=（%.4f，%.4f ，%.4f ） should be zero\n", i, sum.x, sum.y, sum.z);
    }
}

// 使用算好的引力直接更新全体物体速度
__global__ void update_position_and_speed(real* velocity_array, float3* gravity_sum_array, real* position_and_weight, int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程应处理的天体索引
    if (i < N) { // 超过N的不再计算
        int base_index = i << 2;
        real mass = position_and_weight[base_index + 3];
        if (mass != 0.0f) {
            // mass是0的填充点不计算
            velocity_array[base_index] += gravity_sum_array[i].x * TIME_TICK / mass;
            velocity_array[base_index + 1] += gravity_sum_array[i].y * TIME_TICK / mass;
            velocity_array[base_index + 2] += gravity_sum_array[i].z * TIME_TICK / mass;

            position_and_weight[base_index] += velocity_array[base_index] * TIME_TICK;
            position_and_weight[base_index + 1] += velocity_array[base_index + 1] * TIME_TICK;
            position_and_weight[base_index + 2] += velocity_array[base_index + 2] * TIME_TICK;
        }
    }
}

__global__ void update_speed_half(real* velocity_array, float3* gravity_sum_array, real* position_and_weight, int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程应处理的天体索引
    if (i < N) { // 超过N的不再计算
        int base_index = i << 2;
        real mass = position_and_weight[base_index + 3];
        if (mass != 0.0f) {
            // mass是0的填充点不计算
            velocity_array[base_index] += gravity_sum_array[i].x * TIME_TICK / mass / 2;
            velocity_array[base_index + 1] += gravity_sum_array[i].y * TIME_TICK / mass / 2;
            velocity_array[base_index + 2] += gravity_sum_array[i].z * TIME_TICK / mass / 2;
        }
    }
}

__global__ void update_position_complete(real * velocity_array, float3 * gravity_sum_array, real * position_and_weight, int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程应处理的天体索引
    if (i < N) { // 超过N的不再计算
        int base_index = i << 2;
        real mass = position_and_weight[base_index + 3];
        if (mass != 0.0f) {
            // mass是0的填充点不计算
            position_and_weight[base_index] += velocity_array[base_index] * TIME_TICK;
            position_and_weight[base_index + 1] += velocity_array[base_index + 1] * TIME_TICK;
            position_and_weight[base_index + 2] += velocity_array[base_index + 2] * TIME_TICK;
        }
    }
}

/*  #endregion --------------------方法A--------------------  */




/* #region --------------------方法C------------------- */
// 矢量指向为从a指向b，不包含任何的质量
__device__ float3 cal_single_acclerate_without_mass_new(real* a, real* b) {
    // 距离典型值0.01
    //
    
    float compensate = 0.1f;
    float3 dist;
    float3 acc;

    // 计算a到b的距离向量
    dist.x = (b[0] - a[0])* compensate; // 避免距离过近造成数值精度问题，进行补偿
    dist.y = (b[1] - a[1])* compensate;
    dist.z = (b[2] - a[2])* compensate;

    // 计算平方距离，添加一个小值以避免除零错误
    float distSquared = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z + EPSILON;

    // 计算距离的倒数
    float invDist = rsqrtf(distSquared);
    // 得到距离的倒数的三次方
    float invDistCubed = invDist * invDist * invDist*(compensate * compensate);

    // 计算加速度
    acc.x = dist.x * invDistCubed;
    acc.y = dist.y * invDistCubed;
    acc.z = dist.z * invDistCubed;

    return acc;
}
//__device__ float3 cal_single_acclerate_without_mass_new(real* a, real* b) {
//    float3 dist;
//    float3 acc;
//    // a是主物体
//    // b是作用力到a上的物体
//    dist.x = b[0] - a[0];
//    dist.y = b[1] - a[1];
//    dist.z = b[2] - a[2];
//
//    float distSquared = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;  // r^2
//    distSquared += EPSILON; // r^2+epsilon
//    float distance = sqrtf(distSquared); //  r+epsilon
//    float coff = 1/(distance * distance * distance); //     compensate/r^3
//    acc.x = dist.x*coff;
//    acc.y = dist.y*coff;
//    acc.z = dist.z*coff;
//    return acc;
//}

// 对float3类型进行atomic add
__device__ void atomicAdd3(float3* address, float3 val) {
    atomicAdd(&(address->x), val.x);
    atomicAdd(&(address->y), val.y);
    atomicAdd(&(address->z), val.z);
}

// 使用新的方法进行重力计算
// 每列中均进行加速度计算
// 
__global__ void cal_acc_new(float3* acc_array, real* position_and_weight, int block_edge_len, int hori_group_nums, int N) {
    extern __shared__ real sharedMemory[];  // 声明一个动态共享内存，前4*BlockSize个值保存该任务块要用到的质量&位置数组，后4*BlockSize个值保存纵向和 （只保存各个列对应的物体，行对应的物体直接保存在线程内部）
    real* mass = sharedMemory;
    real* column_acc_sum_list = sharedMemory + block_edge_len * 4;
    float3 local_acc_sum = { 0.0f,0.0f,0.0f };
    // size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程应处理的天体索引
    int look_at = 16;
    // if (idx == look_at)printf("noew thread 0\n");
    // 任务块号是blockIdx.x
    // 线程在块内的编号是threadIdx.x
    // 块内线程总数为block_edge_len (= blockDim.x)
    // 根据任务块号和块内编号，加载数据到共享内存
    int group_id_x, group_id_y; // 块在块矩阵中的坐标
    int point_x, point_y, point_y_start; //块的左上角点在任务矩阵中的坐标
    calculate_group_x_y(hori_group_nums, blockIdx.x, &group_id_x, &group_id_y);
    //printf("idx = %d, block_id = %d, group_id_x = %d, group_id_y = %d\n", idx, blockIdx.x,group_id_x, group_id_y);
    calculate_matrix_index(group_id_x, group_id_y, &point_x, &point_y, block_edge_len);// 先找到在总matrix中的起始点，以定位当前处理的行
    //if (blockIdx.x % look_at == 0)printf("idx = %d, block_id = %d, group_id_x=%d, group_id_y=%d, point.x = %d, point.y = %d\n",idx, blockIdx.x, group_id_x, group_id_y, point_x, point_y);
    point_x = point_x + threadIdx.x; // 左上角坐标x->该线程负责计算的真实物体坐标x，同时也是本次计算的行号
    point_y_start = point_y;         // 本次计算的列起始号
    point_y = point_y + threadIdx.x; // 左上角坐标y->该线程负责加载和写入的物体坐标y

    int share_pointer = threadIdx.x*4;  // 等同于 threadIdx.x * 4
    int body_pointer_x = point_x*4;     // 等同于 point_x * 4
    int body_pointer_y = point_y*4;     // 等同于 point_y * 4
    real body_data[4] = {};     // 线程独享的数据
    for (int i = 0;i < 4;i++) { // 每个线程加载到共享内存一个点，block_edge_len个线程一共加载block_edge_len个点的数据
        body_data[i] = position_and_weight[body_pointer_x + i];
        mass[share_pointer + i] = position_and_weight[body_pointer_y + i];
    }
    // 加载结束后,组内同步
    __syncthreads();


    // 正式开始重力/加速度计算
    // 计算任务是（point_x， point_y_start）一直到 （point_x， point_y_start+block_edge_len-1）
    // 也就是线程计算的是一行
    // 行对应的物体数据存在变量body_data里面，列对应的数据直接依次存在mass里面
    int start_local_index = point_x - point_y_start + 1;
    if (start_local_index < 0) {
        start_local_index = 0;
    }

    real* mass_pointer = mass + 4 * start_local_index;
    real* column_sum_pointer = column_acc_sum_list + 4 * start_local_index;
    float3 single_acc;
    for (int i = start_local_index;i < block_edge_len;i++) { // 边算边存，最大利用存储带宽
        single_acc = cal_single_acclerate_without_mass_new(body_data, mass_pointer); // 算出（point_x, point_y+i）的重力值

        local_acc_sum.x += single_acc.x * mass_pointer[3];
        local_acc_sum.y += single_acc.y * mass_pointer[3];
        local_acc_sum.z += single_acc.z * mass_pointer[3];
        // 每列计算结束也不用同步,而是使用原子操作对共享内存进行写入
        // 此处是性能瓶颈
        atomicAdd(&column_sum_pointer[0], single_acc.x * body_data[3]);
        atomicAdd(&column_sum_pointer[1], single_acc.y * body_data[3]);
        atomicAdd(&column_sum_pointer[2], single_acc.z * body_data[3]);
        mass_pointer += 4;
        column_sum_pointer += 4;
    }

    __syncthreads();
    // 计算结束，同步一下，然后将结果保存到全局内存中，每个线程存两次即可。
    // 保存行和
    atomicAdd3(&acc_array[point_x], local_acc_sum);

    // 保存列和
    int column_sum_index = threadIdx.x * 4;
    single_acc.x = column_acc_sum_list[column_sum_index];
    single_acc.y = column_acc_sum_list[column_sum_index + 1];
    single_acc.z = column_acc_sum_list[column_sum_index + 2];
    atomicAdd3(&acc_array[point_y], single_acc);
}

// 在cal_acc_new的基础上的进一步改进
__global__ void cal_acc_advanced(float3* acc_array, real* position_and_weight, int block_edge_len, int hori_group_nums, int N) {
    extern __shared__ real sharedMemory[];  // 声明一个动态共享内存，前4*BlockSize个值保存该任务块要用到的质量&位置数组，后4*BlockSize个值保存纵向和 （只保存各个列对应的物体，行对应的物体直接保存在线程内部）
    real* mass = sharedMemory;
    real* column_acc_sum_list = sharedMemory + block_edge_len * 4;
    float3 local_acc_sum = { 0.0f,0.0f,0.0f };
    // size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程应处理的天体索引
    int look_at = 16;
    // if (idx == look_at)printf("noew thread 0\n");
    // 任务块号是blockIdx.x
    // 线程在块内的编号是threadIdx.x
    // 块内线程总数为block_edge_len (= blockDim.x)
    // 根据任务块号和块内编号，加载数据到共享内存
    int group_id_x, group_id_y; // 块在块矩阵中的坐标
    int point_x, point_y, point_x_start, point_y_start; //块的左上角点在任务矩阵中的坐标
    calculate_group_x_y(hori_group_nums, blockIdx.x, &group_id_x, &group_id_y);
    calculate_matrix_index(group_id_x, group_id_y, &point_x_start, &point_y_start, block_edge_len);// 先找到在总matrix中的起始点，以定位当前处理的行
    point_x = point_x_start + threadIdx.x; // 左上角坐标x->该线程负责计算的真实物体坐标x，同时也是本次计算的行号
    point_y = point_y_start + threadIdx.x; // 左上角坐标y->该线程负责加载和写入的物体坐标y
    if (point_x >= N || point_y >= N) {
        printf("block_id = %d, threadId=%d, group=(%d, %d), point = (%d,%d) out of range\n", blockIdx.x, threadIdx.x, group_id_x, group_id_y, point_x, point_y);
    }

    int share_pointer = threadIdx.x * 4;  // 等同于 threadIdx.x * 4
    int body_pointer_x = point_x * 4;     // 等同于 point_x * 4
    int body_pointer_y = point_y * 4;     // 等同于 point_y * 4
    real body_data[4] = {};     // 线程独享的数据
    for (int i = 0;i < 4;i++) { // 每个线程加载到共享内存一个点，block_edge_len个线程一共加载block_edge_len个点的数据
        body_data[i] = position_and_weight[body_pointer_x + i];
        mass[share_pointer + i] = position_and_weight[body_pointer_y + i];
        column_acc_sum_list[share_pointer + i] = 0;
    }
    // 加载结束后,组内同步
    __syncthreads();

    // 正式开始重力/加速度计算
    // 行对应的物体数据存在局部变量body_data里面，列对应的数据直接依次存在共享内存的mass里面
    int task_len; // 线程需要进行的计算次数
    int current_index = threadIdx.x; // 也就是offset
    if (point_y_start==point_x_start+1){ // 是对角线块
        task_len = block_edge_len - current_index;
    }else {  // 不是对角线块
        task_len = block_edge_len;
    }

    real* mass_pointer;
    real* column_sum_pointer;
    float3 single_acc;
    for (int i = 0;i < task_len; i++) { // 边算边存，最大利用存储带宽
        mass_pointer = mass + 4 * current_index;
        column_sum_pointer = column_acc_sum_list + 4 * current_index;
        int temp_current = point_y_start + current_index; // 应当是当前的y值
        single_acc = cal_single_acclerate_without_mass_new(body_data, mass_pointer); // 算出（point_x, point_y+i）的重力值
        local_acc_sum.x += (single_acc.x * mass_pointer[3]); // （point_y+current_index）号物体对point_x号物体产生的加速度
        local_acc_sum.y += (single_acc.y * mass_pointer[3]);
        local_acc_sum.z += (single_acc.z * mass_pointer[3]);
        // 每列计算结束也不用同步,而是使用原子操作对共享内存进行写入
        // 此处是性能瓶颈,通过调整计算任务顺序，使得各个线程在同一时刻计算的任务不尽相同来减少共享内存冲突。
        atomicAdd(&column_sum_pointer[0], -1*single_acc.x * body_data[3]); // point_x号物体对point_y+current_index号物体产生的加速度
        atomicAdd(&column_sum_pointer[1], -1*single_acc.y * body_data[3]);
        atomicAdd(&column_sum_pointer[2], -1*single_acc.z * body_data[3]);
        
        current_index = (current_index + 1) % block_edge_len;
    }

    __syncthreads();
    // 计算结束，同步一下，然后将结果保存到全局内存中，每个线程存两次即可。
    atomicAdd3(&acc_array[point_x], local_acc_sum);

    // 保存列和
    int column_sum_index = threadIdx.x * 4;
    single_acc.x = column_acc_sum_list[column_sum_index]; // 因为主物体反过来了，所以加速度要反过来
    single_acc.y = column_acc_sum_list[column_sum_index + 1];
    single_acc.z = column_acc_sum_list[column_sum_index + 2];
    atomicAdd3(&acc_array[point_y], single_acc);
}

// 使用计算好的加速度，以及全局内存中的速度和原始位置信息来更新物体位置
__global__ void use_acc_update_position(real* position_and_weight, real* m_deviceVelocity, float3* acc_array) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程应处理的天体索引
    float3 accelerate = acc_array[idx];
    //if (idx == CHECK_IDX) {
    //    printf("%d: acc.x=%.4f, acc.y=%.4f, acc.z=%.4f\n", CHECK_IDX, accelerate.x, accelerate.y, accelerate.z);
    //}
    acc_array[idx] = { 0.0f, 0.0f, 0.0f }; // 用完清空
    // 使用局部变量来减少对全局内存的访问
    real vx = m_deviceVelocity[idx * 4];
    real vy = m_deviceVelocity[idx * 4 + 1];
    real vz = m_deviceVelocity[idx * 4 + 2];

    if (vx >= SPEED_MAX || vy >= SPEED_MAX || vz >= SPEED_MAX) {
        printf("%d overspeed! origin speed=(%.5f, %.5f, %.5f)\n", idx, vx, vy, vz);
    }

    vx += accelerate.x * TIME_TICK;
    vy += accelerate.y * TIME_TICK;
    vz += accelerate.z * TIME_TICK;
    //if (idx == CHECK_IDX) {
    //    printf("%d: (new) speed.x=%.4f, speed.y=%.4f, speed.z=%.4f\n", CHECK_IDX, vx, vy, vz);
    //}

    // 将更新后的速度写回全局内存
    m_deviceVelocity[idx * 4] = vx;
    m_deviceVelocity[idx * 4 + 1] = vy;
    m_deviceVelocity[idx * 4 + 2] = vz;

    // 更新位置
    position_and_weight[idx * 4] += vx * TIME_TICK;
    position_and_weight[idx * 4 + 1] += vy * TIME_TICK;
    position_and_weight[idx * 4 + 2] += vz * TIME_TICK;

    //if (idx == CHECK_IDX) {
    //    printf("%d: pos.x=%.4f, pos.y=%.4f, pos.z=%.4f\n", CHECK_IDX, position_and_weight[idx * 4], position_and_weight[idx * 4 + 1], position_and_weight[idx * 4 + 2]);
    //}

}

/* #endregion --------------------方法C-------------------- */


/*  #region --------------------方法B(简单，但确实有效)--------------------  */

__device__ float3 cal_single_acclerate(real* a, real* b, float3 acc) {
    float3 dist;
    // a是主物体
    // b是作用力到a上的物体
    dist.x = b[0] - a[0];
    dist.y = b[1] - a[1];
    dist.z = b[2] - a[2];
    
    float distSquared = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z;
    distSquared += EPSILON;
    float distance = sqrtf(distSquared);
    float coff = b[3] / (distance* distance* distance);
    acc.x += dist.x * coff;
    acc.y += dist.y * coff;
    acc.z += dist.z * coff;
    return acc;
}

// 调用线程数量 = 物体的数量
// 线程块大小还是1024
__global__ void simple_update_all( real* m_deviceVelocity, real* position_and_weight, int N) {
    extern __shared__ real mass[];
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程应处理的天体索引
    if (idx < N) {
        real main_body[4];
        main_body[0] = position_and_weight[idx*4];
        main_body[1] = position_and_weight[idx*4 + 1];
        main_body[2] = position_and_weight[idx*4 + 2];
        main_body[3] = position_and_weight[idx*4 + 3];
        float3 accelerate = {0.0f, 0.0f, 0.0f};
        size_t  now_pos = 0;
        int k = threadIdx.x;
        float3 last_acc = { 0.0f, 0.0f, 0.0f };
        while (now_pos < 4 * N) {
            // 一个线程加载一个数据，然后计算BLOCK_SIZE个，然后同步一下，然后再加载数据到共享内存，然后再同步一下
            mass[k*4] = position_and_weight[now_pos+ k*4];
            mass[k*4 + 1] = position_and_weight[now_pos + k*4 + 1];
            mass[k*4 + 2] = position_and_weight[now_pos + k*4 + 2];
            mass[k*4 + 3] = position_and_weight[now_pos + k*4 + 3];
            
            __syncthreads(); // 确保大家都加载完成

            for (int i = 0;i < BLOCK_SIZE;i++) {
                accelerate = cal_single_acclerate(main_body, &mass[i * 4], accelerate);
            }
            
            __syncthreads(); // 确保大家都计算完毕

            now_pos += BLOCK_SIZE * 4;
        }

        // 到这里，一个天体的加速度已经计算完成下，然后我们计算更新他的速度


        // 加速器和时间步长
        float ax = accelerate.x * TIME_TICK;
        float ay = accelerate.y * TIME_TICK;
        float az = accelerate.z * TIME_TICK;

        // 更新速度
        m_deviceVelocity[idx * 4] += ax;
        m_deviceVelocity[idx * 4 + 1] += ay;
        m_deviceVelocity[idx * 4 + 2] += az;

        // 将更新后的速度存入局部变量
        float vx = m_deviceVelocity[idx * 4];
        float vy = m_deviceVelocity[idx * 4 + 1];
        float vz = m_deviceVelocity[idx * 4 + 2];


        // 更新位置
        position_and_weight[idx * 4] += vx * TIME_TICK;
        position_and_weight[idx * 4 + 1] += vy * TIME_TICK;
        position_and_weight[idx * 4 + 2] += vz * TIME_TICK;

    }
}

/*
//// 调用线程数量 = 物体的数量
//// 线程块大小还是1024
//__global__ void simple_update_all(real* m_deviceVelocity, real* position_and_weight, int N) {
//    extern __shared__ real mass[];
//    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程应处理的天体索引
//    if (idx < N) {
//        real main_body[4];
//        main_body[0] = position_and_weight[idx * 4];
//        main_body[1] = position_and_weight[idx * 4 + 1];
//        main_body[2] = position_and_weight[idx * 4 + 2];
//        main_body[3] = position_and_weight[idx * 4 + 3];
//        float3 accelerate = { 0.0f, 0.0f, 0.0f };
//        size_t  now_pos = 0;
//        int k = threadIdx.x;
//
//        while (now_pos < 4 * N) {
//            // 一个线程加载一个数据，然后计算BLOCK_SIZE个，然后同步一下，然后再加载数据到共享内存，然后再同步一下
//            mass[k * 4] = position_and_weight[now_pos + k * 4];
//            mass[k * 4 + 1] = position_and_weight[now_pos + k * 4 + 1];
//            mass[k * 4 + 2] = position_and_weight[now_pos + k * 4 + 2];
//            mass[k * 4 + 3] = position_and_weight[now_pos + k * 4 + 3];
//
//            __syncthreads(); // 确保大家都加载完成
//
//            for (int i = 0;i < BLOCK_SIZE;i++) {
//                if (idx == CHECK_IDX && now_pos == 0 && i < 10) {
//                    float3 acc_before = accelerate;
//                    accelerate = cal_single_acclerate(main_body, &mass[i * 4], accelerate);
//                    printf("(%d,%d) acc.x=%.4f, acc.y=%.4f, acc.z=%.4f\n", CHECK_IDX, now_pos + i, accelerate.x - acc_before.x, accelerate.y - acc_before.y, accelerate.z - acc_before.z);
//                }
//                else {
//                    accelerate = cal_single_acclerate(main_body, &mass[i * 4], accelerate);
//                }
//            }
//
//            __syncthreads(); // 确保大家都计算完毕
//            now_pos += BLOCK_SIZE * 4;
//        }
//        // printf("now_pos_final=%d\n", now_pos);
//
//        // 到这里，一个天体的加速度已经计算完成下，然后我们计算更新他的速度
//
//        // float abcd = accelerate.y;
//
//        //m_deviceVelocity[idx * 4] += accelerate.x * TIME_TICK;
//        //m_deviceVelocity[idx*4+1] += accelerate.y * TIME_TICK;
//        //m_deviceVelocity[idx*4+2] += accelerate.z * TIME_TICK;
//
//        //// 最后更新他的位置
//        //position_and_weight[idx*4] += m_deviceVelocity[idx*4] * TIME_TICK;
//        //position_and_weight[idx*4+1] += m_deviceVelocity[idx*4+1] * TIME_TICK;
//        //position_and_weight[idx*4+2] += m_deviceVelocity[idx*4+2] * TIME_TICK;
//        if (idx == CHECK_IDX) {
//            printf("%d: acc.x=%.4f, acc.y=%.4f, acc.z=%.4f\n", CHECK_IDX, accelerate.x, accelerate.y, accelerate.z);
//        }
//
//        // 加速器和时间步长
//        float ax = accelerate.x * TIME_TICK;
//        float ay = accelerate.y * TIME_TICK;
//        float az = accelerate.z * TIME_TICK;
//
//        // 更新速度
//        m_deviceVelocity[idx * 4] += ax;
//        m_deviceVelocity[idx * 4 + 1] += ay;
//        m_deviceVelocity[idx * 4 + 2] += az;
//
//        // 将更新后的速度存入局部变量
//        float vx = m_deviceVelocity[idx * 4];
//        float vy = m_deviceVelocity[idx * 4 + 1];
//        float vz = m_deviceVelocity[idx * 4 + 2];
//
//        if (idx == CHECK_IDX) {
//            printf("%d: speed.x=%.4f, speed.y=%.4f, speed.z=%.4f\n", CHECK_IDX, vx, vy, vz);
//        }
//
//        // 更新位置
//        position_and_weight[idx * 4] += vx * TIME_TICK;
//        position_and_weight[idx * 4 + 1] += vy * TIME_TICK;
//        position_and_weight[idx * 4 + 2] += vz * TIME_TICK;
//
//        if (idx == CHECK_IDX) {
//            printf("%d: pos.x=%.4f, pos.y=%.4f, pos.z=%.4f\n", CHECK_IDX, position_and_weight[idx * 4], position_and_weight[idx * 4 + 1], position_and_weight[idx * 4 + 2]);
//        }
//    }
//}


// 更高精度的update
//__global__ void simple_update_all(real* m_deviceVelocity, real* position_and_weight, int N) {
//    extern __shared__ real mass[];
//    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if (idx < N) {
//        real main_body[4];
//        main_body[0] = position_and_weight[idx * 4];
//        main_body[1] = position_and_weight[idx * 4 + 1];
//        main_body[2] = position_and_weight[idx * 4 + 2];
//        main_body[3] = position_and_weight[idx * 4 + 3];
//        float3 accelerate = { 0.0f, 0.0f, 0.0f };
//        size_t now_pos = 0;
//        int k = threadIdx.x;
//
//        // First half-step velocity update
//        while (now_pos < 4 * N) {
//            mass[k * 4] = position_and_weight[now_pos + k * 4];
//            mass[k * 4 + 1] = position_and_weight[now_pos + k * 4 + 1];
//            mass[k * 4 + 2] = position_and_weight[now_pos + k * 4 + 2];
//            mass[k * 4 + 3] = position_and_weight[now_pos + k * 4 + 3];
//
//            now_pos += BLOCK_SIZE * 4;
//            __syncthreads();
//
//            for (int i = 0; i < BLOCK_SIZE; i++) {
//                accelerate = cal_single_acclerate(main_body, &mass[i * 4], accelerate);
//            }
//            __syncthreads();
//        }
//
//        // Update velocity with first half-step acceleration
//        m_deviceVelocity[idx * 4] += accelerate.x * (TIME_TICK / 2.0f);
//        m_deviceVelocity[idx * 4 + 1] += accelerate.y * (TIME_TICK / 2.0f);
//        m_deviceVelocity[idx * 4 + 2] += accelerate.z * (TIME_TICK / 2.0f);
//
//        // Update position
//        position_and_weight[idx * 4] += m_deviceVelocity[idx * 4] * TIME_TICK;
//        position_and_weight[idx * 4 + 1] += m_deviceVelocity[idx * 4 + 1] * TIME_TICK;
//        position_and_weight[idx * 4 + 2] += m_deviceVelocity[idx * 4 + 2] * TIME_TICK;
//
//        // Recompute acceleration for the second half-step velocity update
//        accelerate = { 0.0f, 0.0f, 0.0f };
//        now_pos = 0;
//
//        // Synchronize threads to ensure all threads have updated positions
//        __syncthreads();
//
//        main_body[0] = position_and_weight[idx * 4];
//        main_body[1] = position_and_weight[idx * 4 + 1];
//        main_body[2] = position_and_weight[idx * 4 + 2];
//        main_body[3] = position_and_weight[idx * 4 + 3];
//
//        while (now_pos < 4 * N) {
//            mass[k * 4] = position_and_weight[now_pos + k * 4];
//            mass[k * 4 + 1] = position_and_weight[now_pos + k * 4 + 1];
//            mass[k * 4 + 2] = position_and_weight[now_pos + k * 4 + 2];
//            mass[k * 4 + 3] = position_and_weight[now_pos + k * 4 + 3];
//
//            now_pos += BLOCK_SIZE * 4;
//            __syncthreads();
//
//            for (int i = 0; i < BLOCK_SIZE; i++) {
//                accelerate = cal_single_acclerate(main_body, &mass[i * 4], accelerate);
//            }
//            __syncthreads();
//        }
//
//        // Update velocity with second half-step acceleration
//        m_deviceVelocity[idx * 4] += accelerate.x * (TIME_TICK / 2.0f);
//        m_deviceVelocity[idx * 4 + 1] += accelerate.y * (TIME_TICK / 2.0f);
//        m_deviceVelocity[idx * 4 + 2] += accelerate.z * (TIME_TICK / 2.0f);
//    }
//}
*/
/* #endregion --------------------方法B-------------------- */



/* #region --------------------串行计算方法--------------------*/

// 单线程串行版本的核函数
__global__ void single_thread_update_all(real* m_deviceVelocity, real* position_and_weight, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int idx = 0; idx < N; idx++) {
            real main_body[4];
            main_body[0] = position_and_weight[idx * 4];
            main_body[1] = position_and_weight[idx * 4 + 1];
            main_body[2] = position_and_weight[idx * 4 + 2];
            main_body[3] = position_and_weight[idx * 4 + 3];
            float3 accelerate = { 0.0f, 0.0f, 0.0f };

            for (int j = 0; j < N; j++) {
                if (idx != j) {
                    real other_body[4];
                    other_body[0] = position_and_weight[j * 4];
                    other_body[1] = position_and_weight[j * 4 + 1];
                    other_body[2] = position_and_weight[j * 4 + 2];
                    other_body[3] = position_and_weight[j * 4 + 3];
                    accelerate = cal_single_acclerate(main_body, other_body, accelerate);
                }
            }

            // 更新速度
            m_deviceVelocity[idx * 4] += accelerate.x * TIME_TICK;
            m_deviceVelocity[idx * 4 + 1] += accelerate.y * TIME_TICK;
            m_deviceVelocity[idx * 4 + 2] += accelerate.z * TIME_TICK;

            // 更新位置
            position_and_weight[idx * 4] += m_deviceVelocity[idx * 4] * TIME_TICK;
            position_and_weight[idx * 4 + 1] += m_deviceVelocity[idx * 4 + 1] * TIME_TICK;
            position_and_weight[idx * 4 + 2] += m_deviceVelocity[idx * 4 + 2] * TIME_TICK;
        }
    }
}

/* #endregion --------------------串行计算方法--------------------*/


__global__ void printFirstDataPoint(real* data) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {  // 确保只有一个线程执行打印
        for (int i = 0;i < 5;i++) {
            printf("No.%d data point: x=%f, y=%f, z=%f, w=%f\n", i, data[0+4*i], data[1+4*i], data[2+4*i], data[3+4*i]);
        }
    }
}

__global__ void findMaxAbsValues(real* data, int numPoints, real* maxValues) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {  // 确保只有一个线程执行
        real maxX = 0;
        real maxY = 0;
        real maxZ = 0;
        real maxW = 0;

        // 遍历所有点
        for (int i = 0; i < numPoints; i++) {
            real x = fabs(data[4 * i]);     // x坐标的绝对值
            real y = fabs(data[4 * i + 1]); // y坐标的绝对值
            real z = fabs(data[4 * i + 2]); // z坐标的绝对值
            real w = fabs(data[4 * i + 3]); // w坐标的绝对值

            // 更新最大值
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
            if (z > maxZ) maxZ = z;
            if (w > maxW) maxW = w;
        }

        // 将结果存储在maxValues数组中
        maxValues[0] = maxX;
        maxValues[1] = maxY;
        maxValues[2] = maxZ;
        maxValues[3] = maxW;
    }
}

//__global__ void


void resizeWindowCallback(GLFWwindow* window, int width, int height)
{
    // 更新 OpenGL 视口大小
    glViewport(0, 0, width, height);
}

void load_data(int choice) {
    if (choice == 0) { // 一个简单的旋转星系
        loadTipsyFile("./data/galaxy_20K.bin");
        scaleFactors[0] = 200;
        scaleFactors[1] = 120;
        scaleFactors[2] = 200;
    }
    else if (choice == 1) { // 两个旋转星系相撞
        loadTabFile("./data/dubinski.tab");
        scaleFactors[0] = 100;
        scaleFactors[1] = 100;
        scaleFactors[2] = 100;
    }
    else if (choice == 2) { // 粒子数量众多的一个简单中心星系
        loadTabFile("./data/tab65536.tab");
        scaleFactors[0] = 50;
        scaleFactors[1] = 50;
        scaleFactors[2] = 50;
    }
    else if (choice == 3) { // 星团爆炸
        loadDatFile("./data/stars.dat");
        scaleFactors[0] = 400;
        scaleFactors[1] = 400;
        scaleFactors[2] = 400;
        cameraDistance = 3;
    }
    else if (choice == 4) { // 十帧电竞
        loadDatFile("./data/k17c.snap");
        scaleFactors[0] = 300;
        scaleFactors[1] = 300;
        scaleFactors[2] = 300;
    }
    else if (choice == 5) {
        loadDatFile("./data/k17hp.snap");
        scaleFactors[0] = 300;
        scaleFactors[1] = 300;
        scaleFactors[2] = 300;
    }
}


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (yoffset < 0) {
        cameraDistance *= 1.1f;
    }
    else if (yoffset > 0) {
        cameraDistance *= 0.9f;
    }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            isMousePressed = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        }
        else if (action == GLFW_RELEASE) {
            isMousePressed = false;
        }
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (isMousePressed) {
        double deltaX = xpos - lastMouseX;
        double deltaY = lastMouseY - ypos;

        float sensitivity = 0.1f; // 鼠标移动的灵敏度

        sphereCoords.x -= deltaX * sensitivity;
        sphereCoords.y -= deltaY * sensitivity;

        // 限制phi的范围在-90到90度之间
        sphereCoords.y = glm::clamp(sphereCoords.y, -89.0f, 89.0f);

        lastMouseX = xpos;
        lastMouseY = ypos;
    }
}

void updateCameraPos() {
    float theta = glm::radians(sphereCoords.x);
    float phi = glm::radians(sphereCoords.y);

    float x = cos(phi) * sin(theta);
    float y = sin(phi);
    float z = cos(phi) * cos(theta);

    cameraPos = glm::vec3(x, y, z);
}



// cuda中分配的数组有：
// 全局速度数组
// 全局位置重量数组
// 全局加速度数组
int main(int argc, char** argv) {

    int DATA_ID = DEFAULT_DATASET;
    if (argc > 1) {
        try {
            DATA_ID = std::stoi(argv[1]); // 将第一个命令行参数转为整数并赋值给value
        }
        catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << argv[1] << " is not a valid number." << std::endl;
            return 1;
        }
        catch (const std::out_of_range& e) {
            std::cerr << "Out of range: " << argv[1] << " is out of range for an int." << std::endl;
            return 1;
        }
    }
    if (DATA_ID > 5) {
        printf("DATA_ID should be 0-5.\n");
        exit(0);
    }

    // 初始化和设置窗口
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(2400, 1800, "Point Sprites", nullptr, nullptr);
    if (!window) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, resizeWindowCallback);
    // 初始化glad
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    // 设置键盘回调函数
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    // 设置和编译着色器
    GLuint shaderProgram = loadShader("vertex_shader.glsl", "fragment_shader.glsl");

    // 读入全部的顶点数据
    load_data(DATA_ID);
    // 调试选项
    if (m_numBodies > MAX_BODIES){
        m_numBodies = MAX_BODIES;
    }
    if (real_body_nums > MAX_BODIES) {
        real_body_nums = MAX_BODIES;
    }
    

    printf("read star data done.\n");
    printf("num of Bodies = %d\n", m_numBodies);

    
    // 映射初始化好的buf资源
    cudaGraphicsMapResources(1, &m_pGRes[0], 0); //第一个参数是映射的buf数量，第二个参数是资源的指针的数组， 第三个参数是流
    void* devPtr;
    size_t size;
    cudaGraphicsResourceGetMappedPointer(&devPtr, &size, m_pGRes[0]);
    printf("initial_len: %d Bytes\n", size);
    real* devMaxValues;
    cudaMalloc(&devMaxValues, 4 * sizeof(real));
    printFirstDataPoint <<<1, 1>>> ((real*)devPtr);
    cudaDeviceSynchronize();
    real maxValues[4];
    findMaxAbsValues <<<1, 1 >>> ((real*)devPtr, 20225, devMaxValues);
    cudaMemcpy(maxValues, devMaxValues, 4 * sizeof(real), cudaMemcpyDeviceToHost);
    printf("Max absolute values 1: X=%f, Y=%f, Z=%f, W=%f\n", maxValues[0], maxValues[1], maxValues[2], maxValues[3]);
    cudaGraphicsUnmapResources(1, &m_pGRes[0], 0); // 使用完毕后释放资源，交还给openGL
    printf("really body counts = %d\n", real_body_nums);
    printf(" return control to openGL\n");


    int body_num = m_numBodies;
    int all_task_num = m_numBodies * (m_numBodies - 1) / 2;
    int task_block_edge_len = BLOCK_SIZE; // 任务块大小=block内线程数量
    int hori_group_nums = (body_num - 2) / task_block_edge_len + 1; // 多减了一个1是因为如果有N个物品，则对应一行只有N-1个任务，(0,0)不计算
    int block_num = hori_group_nums * (hori_group_nums + 1) / 2; // 计算引力矩阵的线程数量

    // 分配引力数组
    //float3* gravity_array; // 各个天体之间的引力matrix拉伸成一个数组
    //cudaError_t status = cudaMalloc((void**)&gravity_array, all_task_num * sizeof(float3));

    float3* gravity_sum_array;
    cudaError_t status = cudaMalloc((void**)&gravity_sum_array, m_numBodies * sizeof(float3));

    if (status != cudaSuccess) {
        printf("gravity_sum_array CUDA malloc failed: %s\n", cudaGetErrorString(status));
        return -1;
    }

    status = cudaMemset(gravity_sum_array, 0, m_numBodies * sizeof(float3));
    if (status != cudaSuccess) {
        printf("gravity_sum_array CUDA memset failed: %s\n", cudaGetErrorString(status));
        return -1;
    }



    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_pbo[m_currentRead]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(real), (void*)0); // 位置属性
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(real), (void*)(3 * sizeof(real))); // 质量属性，假设为点大小
    glEnableVertexAttribArray(1);
    // 解绑操作，确保不再有任何顶点缓冲区（VBO）和顶点数组对象（VAO）被当前的OpenGL上下文绑定
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);


    void* position_and_weight;
    size_t resourse_size;
    int share_memory_size = BLOCK_SIZE * sizeof(real) * 4;//每个天体需要4个real类型长度,每个block存task_block_edge_len个点
    int count = 0;
    float frame_count = 0;
    glfwSetScrollCallback(window, scroll_callback);
    GLint scaleFactorsLocation = glGetUniformLocation(shaderProgram, "scaleFactors");
    GLint viewLocation = glGetUniformLocation(shaderProgram, "view");
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)2400 / (float)1800, 0.1f, 100.0f);
    GLint projectionLocation = glGetUniformLocation(shaderProgram, "projection");

    // 渲染循环
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        auto frame_start = std::chrono::high_resolution_clock::now();

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // 清理缓冲区
        // printf("running %d\n", count);

        auto start_compute = std::chrono::high_resolution_clock::now();
        #if VERSION == 0
            // 简单模拟法
            cudaGraphicsMapResources(1, &m_pGRes[0], 0);
            cudaGraphicsResourceGetMappedPointer(&position_and_weight, &resourse_size, m_pGRes[0]);
            cal_gravity_new << <block_num, task_block_edge_len, share_memory_size >> > (gravity_array, (real*)position_and_weight, task_block_edge_len, hori_group_nums, m_numBodies);
            cudaDeviceSynchronize();
            auto compute_point_1 = std::chrono::high_resolution_clock::now();
            add_up_gravity << < (m_numBodies - 1) / 256 + 1, 256 >> > (gravity_array, gravity_sum_array, m_numBodies);
            cudaDeviceSynchronize();
            auto compute_point_2 = std::chrono::high_resolution_clock::now();
            update_position_and_speed << <(m_numBodies - 1) / 512 + 1, 512 >> > (m_deviceVelocity, gravity_sum_array, (real*)position_and_weight, m_numBodies);
            cudaDeviceSynchronize();
            cudaGraphicsUnmapResources(1, &m_pGRes[0], 0); // 使用完毕后释放资源，交还给openGL
        #elif VERSION == 1
            // 两步模拟法
            cudaGraphicsMapResources(1, &m_pGRes[0], 0);
            cudaGraphicsResourceGetMappedPointer(&position_and_weight, &resourse_size, m_pGRes[0]);
            cal_gravity << <block_num, task_block_edge_len, share_memory_size >> > (gravity_array, (real*)position_and_weight, task_block_edge_len, hori_group_nums, m_numBodies);
            cudaDeviceSynchronize();
            add_up_gravity << < (m_numBodies - 1) / 1024 + 1, 1024 >> > (gravity_array, gravity_sum_array, m_numBodies);
            cudaDeviceSynchronize();
            update_speed_half << <(m_numBodies - 1) / 1024 + 1, 1024 >> > (m_deviceVelocity, gravity_sum_array, (real*)position_and_weight, m_numBodies);
            cudaDeviceSynchronize();
            update_position_complete << <(m_numBodies - 1) / 1024 + 1, 1024 >> > (m_deviceVelocity, gravity_sum_array, (real*)position_and_weight, m_numBodies);
            cudaDeviceSynchronize();
            cal_gravity << <block_num, task_block_edge_len, share_memory_size >> > (gravity_array, (real*)position_and_weight, task_block_edge_len, hori_group_nums, m_numBodies);
            cudaDeviceSynchronize();
            add_up_gravity << < (m_numBodies - 1) / 1024 + 1, 1024 >> > (gravity_array, gravity_sum_array, m_numBodies);
            cudaDeviceSynchronize();
            update_speed_half << <(m_numBodies - 1) / 1024 + 1, 1024 >> > (m_deviceVelocity, gravity_sum_array, (real*)position_and_weight, m_numBodies);
            cudaDeviceSynchronize();
            cudaGraphicsUnmapResources(1, &m_pGRes[0], 0); // 使用完毕后释放资源，交还给openGL
        #elif VERSION == 2
            // 大道至简法
            cudaGraphicsMapResources(1, &m_pGRes[0], 0);
            cudaGraphicsResourceGetMappedPointer(&position_and_weight, &resourse_size, m_pGRes[0]);
            simple_update_all << < (m_numBodies - 2) / BLOCK_SIZE + 1, BLOCK_SIZE, share_memory_size >> > ((real*)m_deviceVelocity, (real*)position_and_weight, m_numBodies-1);
            cudaDeviceSynchronize();
            auto compute_point_1 = std::chrono::high_resolution_clock::now();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("simple_update_all: CUDA Error: %s\n", cudaGetErrorString(err));
            }
            cudaGraphicsUnmapResources(1, &m_pGRes[0], 0); // 使用完毕后释放资源，交还给openGL
            auto compute_point_2 = std::chrono::high_resolution_clock::now();
        
        #elif VERSION == 3
            // 串行方法
            cudaGraphicsMapResources(1, &m_pGRes[0], 0);
            cudaGraphicsResourceGetMappedPointer(&position_and_weight, &resourse_size, m_pGRes[0]);
            single_thread_update_all <<< 1,1 >>> ((real*)m_deviceVelocity, (real*)position_and_weight, m_numBodies-1);
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("simple_update_all: CUDA Error: %s\n", cudaGetErrorString(err));
            }
            cudaGraphicsUnmapResources(1, &m_pGRes[0], 0); // 使用完毕后释放资源，交还给openGL
        #elif VERSION == 4
            // 优化后的自定义方法
            cudaGraphicsMapResources(1, &m_pGRes[0], 0);
            cudaGraphicsResourceGetMappedPointer(&position_and_weight, &resourse_size, m_pGRes[0]);

            // 将所有的加速度计算好放在acc_array中（为了方便使用各种方法进行运行，这里使用了gravity_array这个变量名）
            cal_acc_new<<<block_num, task_block_edge_len, share_memory_size*2 >>> (gravity_array, (real*)position_and_weight, task_block_edge_len, hori_group_nums, m_numBodies); // 这里名为gravity_array，实为acc_array
            cudaDeviceSynchronize();
            auto compute_point_1 = std::chrono::high_resolution_clock::now();
            // 使用计算好的acc来更新位置
            use_acc_update_position << <(m_numBodies - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > ((real*)position_and_weight, (real*)m_deviceVelocity, gravity_array);
            cudaDeviceSynchronize();
            auto compute_point_2 = std::chrono::high_resolution_clock::now();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("simple_update_all: CUDA Error: %s\n", cudaGetErrorString(err));
            }
            cudaGraphicsUnmapResources(1, &m_pGRes[0], 0); // 使用完毕后释放资源，交还给openGL
        #elif VERSION == 5
            // 第二次优化后的自定义方法
            cudaGraphicsMapResources(1, &m_pGRes[0], 0);
            cudaGraphicsResourceGetMappedPointer(&position_and_weight, &resourse_size, m_pGRes[0]);
            // 将所有的加速度计算好放在acc_array中（为了方便使用各种方法进行运行，这里使用了gravity_array这个变量名）
            // 最大内存访问为：
            // printf("Block num=%d, should be 3160\n", block_num);
            cal_acc_advanced << <block_num, BLOCK_SIZE, share_memory_size * 2 >> > (gravity_sum_array, (real*)position_and_weight, BLOCK_SIZE, hori_group_nums, m_numBodies); // 这里名为gravity_array，实为acc_array
            
            cudaDeviceSynchronize();
            auto compute_point_1 = std::chrono::high_resolution_clock::now();
            // 使用计算好的acc来更新位置
            use_acc_update_position << < (real_body_nums-1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > ((real*)position_and_weight, (real*)m_deviceVelocity, gravity_sum_array);
            cudaDeviceSynchronize();
            auto compute_point_2 = std::chrono::high_resolution_clock::now();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("simple_update_all: CUDA Error: %s\n", cudaGetErrorString(err));
            }
            cudaGraphicsUnmapResources(1, &m_pGRes[0], 0); // 使用完毕后释放资源，交还给openGL
        #elif VERSION == 6
            // 最后一次DEBUG!!!!!不成功便成仁
            cudaGraphicsMapResources(1, &m_pGRes[0], 0);
            cudaGraphicsResourceGetMappedPointer(&position_and_weight, &resourse_size, m_pGRes[0]);
            // 将所有的加速度计算好放在acc_array中（为了方便使用各种方法进行运行，这里使用了gravity_array这个变量名）
            // 最大内存访问为：
            printf("Block num=%d, should be 3160\n", block_num);
            cal_acc_debug << <block_num, BLOCK_SIZE, share_memory_size * 2 >> > (gravity_sum_array, (real*)position_and_weight, BLOCK_SIZE, hori_group_nums, m_numBodies); // 这里名为gravity_array，实为acc_array

            cudaDeviceSynchronize();
            auto compute_point_1 = std::chrono::high_resolution_clock::now();
            // 使用计算好的acc来更新位置
            use_acc_update_position << < (real_body_nums - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > ((real*)position_and_weight, (real*)m_deviceVelocity, gravity_sum_array);
            cudaDeviceSynchronize();
            auto compute_point_2 = std::chrono::high_resolution_clock::now();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("simple_update_all: CUDA Error: %s\n", cudaGetErrorString(err));
            }
            cudaGraphicsUnmapResources(1, &m_pGRes[0], 0); // 使用完毕后释放资源，交还给openGL

        #endif

        //auto end_compute = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double, std::milli> compute_time = end_compute - start_compute;
        //std::cout << "Total Compute Time:" << compute_time.count() << " ms" << std::endl;
        //std::chrono::duration<double, std::milli> compute_time_1 = compute_point_1 - start_compute;
        //std::cout << "start-point1 Time:" << compute_time_1.count() << " ms" << std::endl;
        //std::chrono::duration<double, std::milli> compute_time_2 = compute_point_2 - compute_point_1;
        //std::cout << "point1-point2 Time:" << compute_time_2.count() << " ms" << std::endl;
        //std::chrono::duration<double, std::milli> compute_time_3 = end_compute - compute_point_2;
        //std::cout << "point2-end Time:" << compute_time_3.count() << " ms" << std::endl;

        // 绘制点精灵
        auto start_render = std::chrono::high_resolution_clock::now();
        glUseProgram(shaderProgram);
        glUniform3fv(scaleFactorsLocation, 1, scaleFactors);
        // 传递当前视角矩阵到着色器
        updateCameraPos();
        glm::mat4 view = glm::lookAt(cameraPos * cameraDistance,   //摄像机的位置向量
                                     cameraTarget,   //目标位置向量
                                     cameraUp);  //上向量
        glUniformMatrix4fv(viewLocation, 1, GL_FALSE, glm::value_ptr(view));
        // 把投影矩阵传给着色器
        glUniformMatrix4fv(projectionLocation, 1, GL_FALSE, glm::value_ptr(projection));
        // 重新绑定VAO和VBO
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_pbo[0]);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glDrawArrays(GL_POINTS, 0, real_body_nums); // 只显示真实的点，填充的点不渲染
        glBindVertexArray(0);
        //auto end_render = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double, std::milli> render_time = end_render - start_render;
        //std::cout << "render time:" << render_time.count() << " ms" << std::endl;
        auto frame_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> frame_time = frame_end - frame_start;
        double frame_time_ms = frame_time.count();
        double fps = 1000.0 / frame_time_ms;

        // 更新窗口标题
        count++;
        frame_count += frame_time_ms;
        if (count == 100) {
            
            float average_frame = frame_count / count;
            count = 0;
            frame_count = 0;
            std::stringstream ss;
            ss << "Frame Time: " << average_frame << " ms | FPS: " << fps;
            glfwSetWindowTitle(window, ss.str().c_str());
        }
        glfwSwapBuffers(window);
        glfwPollEvents();
        // printf("------------------------------\n");
        // std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    // 清理资源
    glDeleteVertexArrays(1, &VAO);
    glDeleteProgram(shaderProgram);
    cudaFree(m_deviceVelocity);
    cudaFree(gravity_sum_array);
    cudaGraphicsUnregisterResource(m_pGRes[0]);
    cudaGraphicsUnregisterResource(m_pGRes[1]);
    glDeleteBuffers(2, m_pbo);

    glfwTerminate();
    return 0;
}