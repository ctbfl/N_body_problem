#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>    // For isnan()

typedef float real;

struct Header {
    double time;
    int nbodies;
    int ndimension;
    int nsph;
    int ndark;
    int nstar;
};

struct DarkParticle {
    real mass;
    real pos[3];
    real vel[3];
    real eps;
    int phi;
};

struct StarParticle {
    real mass;
    real pos[3];
    real vel[3];
    real metals;
    real tform;
    real eps;
    int phi;
};

void readTipsyFile(const std::string& inputFileName, const std::string& outputFileName) {
    std::ifstream input(inputFileName, std::ios::in | std::ios::binary);
    std::ofstream output(outputFileName, std::ios::out);
    if (!input.is_open()) {
        std::cerr << "Could not open the binary file: " << inputFileName << std::endl;
        return;
    }
    if (!output.is_open()) {
        std::cerr << "Could not open the CSV file: " << outputFileName << std::endl;
        return;
    }

    Header header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));

    output << "id,x,y,z,mass,vx,vy,vz,eps\n"; // CSV Header

    int nTotal = header.nbodies;
    for (int id = 0; id < nTotal; ++id) {
        real pos[4], vel[4];  // w-component for pos is mass, for vel is eps
        int phi;
        if (id < header.ndark) {
            // Read dark particle data
            DarkParticle particle;
            input.read(reinterpret_cast<char*>(&particle), sizeof(particle));
            pos[0] = particle.pos[0];
            pos[1] = particle.pos[1];
            pos[2] = particle.pos[2];
            pos[3] = particle.mass;
            vel[0] = particle.vel[0];
            vel[1] = particle.vel[1];
            vel[2] = particle.vel[2];
            vel[3] = particle.eps;
            phi = particle.phi;
        } else {
            // Read star particle data
            StarParticle particle;
            input.read(reinterpret_cast<char*>(&particle), sizeof(particle));
            pos[0] = particle.pos[0];
            pos[1] = particle.pos[1];
            pos[2] = particle.pos[2];
            pos[3] = particle.mass;
            vel[0] = particle.vel[0];
            vel[1] = particle.vel[1];
            vel[2] = particle.vel[2];
            vel[3] = particle.eps;
            phi = particle.phi;
        }
        output << id << "," << pos[0] << "," << pos[1] << "," << pos[2] << "," << pos[3] << ","
               << vel[0] << "," << vel[1] << "," << vel[2] << "," << vel[3] << "\n";
    }

    input.close();
    output.close();
}

int main() {
    readTipsyFile("data/galaxy_20K.bin", "data/galaxy_20k.csv");
    return 0;
}