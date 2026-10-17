#include "magics.h"

uint64_t ROOK_MAGICS[64] = {
    0x80008334204000ULL,
    0x40001000200041ULL,
    0x8100102000440900ULL,
    0x500210010002428ULL,
    0x100080010050002ULL,
    0x4100080100020400ULL,
    0x200380082000704ULL,
    0x4080002500085080ULL,
    0x260800080400020ULL,
    0x2019002040008508ULL,
    0x5a04801000200681ULL,
    0x881001000090020ULL,
    0x60200040a001120ULL,
    0x6182001004080200ULL,
    0x4000401923008ULL,
    0x214800100004080ULL,
    0x2000208000804001ULL,
    0x1000818040002000ULL,
    0x8420808010002002ULL,
    0x6024220040081200ULL,
    0x4008004800800ULL,
    0x2001010008020400ULL,
    0x34c1040002181091ULL,
    0x220020408104ULL,
    0x401080002089ULL,
    0x1220008080304000ULL,
    0x1400100080200082ULL,
    0x90001080080280ULL,
    0x808c000808008040ULL,
    0x802000200081004ULL,
    0x810400304812ULL,
    0x804200010084ULL,
    0x9000400020800082ULL,
    0x8260100020400040ULL,
    0x4810402001001104ULL,
    0x421211202000a40ULL,
    0x1028080081800400ULL,
    0x400800200800400ULL,
    0x5820124001008ULL,
    0x1000164882001104ULL,
    0x8857040018000ULL,
    0x8000402010004000ULL,
    0x8205220080460012ULL,
    0xb0080010008080ULL,
    0x808040008008080ULL,
    0x2010020004008080ULL,
    0x202004104820008ULL,
    0x12000a041020004ULL,
    0x8900410020800100ULL,
    0x900802042010200ULL,
    0xc401082042200ULL,
    0x800091001002500ULL,
    0x8008004000880ULL,
    0x200040080020080ULL,
    0x100208010400ULL,
    0x1312004430810a00ULL,
    0x80320801a004102ULL,
    0x1005022810408202ULL,
    0x9005100a41002001ULL,
    0x2000100105002009ULL,
    0x11200a004908802ULL,
    0x301000400020801ULL,
    0x2200080090010204ULL,
    0x880088031040442ULL
};

uint64_t BISHOP_MAGICS[64] = {
    0xa010220208021011ULL,
    0x1a0040102002804ULL,
    0x4022180100241100ULL,
    0x8114405081020010ULL,
    0x202021001800503ULL,
    0x8202022260044000ULL,
    0x101c82868088428ULL,
    0x1211400844000ULL,
    0x100091028801040cULL,
    0x240108010908ULL,
    0x881200420c04ULL,
    0xe00080604400002ULL,
    0x2c02911040000080ULL,
    0x40682a0131084020ULL,
    0x808028801905000ULL,
    0x2090604c02080300ULL,
    0x4044003021880100ULL,
    0x4100250020200ULL,
    0x40504202004100ULL,
    0x8220404009010ULL,
    0x11000820080001ULL,
    0x200802300a00110ULL,
    0x1381022801480bULL,
    0x680a000100a08400ULL,
    0x8220404841000ULL,
    0x8200002023220ULL,
    0x12080801080020ULL,
    0x260120010400440ULL,
    0x8301001001004001ULL,
    0x408244022004200ULL,
    0x8044021010890ULL,
    0x814011004490091ULL,
    0x108282816242100ULL,
    0x1008018810300a20ULL,
    0xc00400240c120400ULL,
    0x20084080080ULL,
    0x8008060400001100ULL,
    0x8020c0041080800ULL,
    0x28900d0240021233ULL,
    0x14042228048080ULL,
    0x1041082010658402ULL,
    0x822c421010008459ULL,
    0x1110808044100ULL,
    0x80011144041800ULL,
    0x240201208800400ULL,
    0x536048102002104ULL,
    0x11102400c4101080ULL,
    0x221009020c280080ULL,
    0x84020150080000ULL,
    0x4600c4200050ULL,
    0x18048181104ULL,
    0x400000084240400ULL,
    0x80014a0821010000ULL,
    0x120444d102020002ULL,
    0x4010050104040004ULL,
    0x2204282818408080ULL,
    0x100404218200200ULL,
    0x9023134802080cULL,
    0x25000108411001ULL,
    0x2028010c4840400ULL,
    0x40022007a10a401ULL,
    0x410660e05ULL,
    0x8240090810041260ULL,
    0x2200801010020ULL
};

uint64_t ROOK_MASKS[64];
uint64_t BISHOP_MASKS[64];
int ROOK_SHIFTS[64];
int BISHOP_SHIFTS[64];
uint64_t* ROOK_ATTACKS[64];
uint64_t* BISHOP_ATTACKS[64];
static uint64_t rook_table[102400];
static uint64_t bishop_table[5248];

static uint64_t calc_rook_mask(int sq) { uint64_t r=0; int rr=sq/8, ff=sq%8; for(int i=rr+1; i<=6; i++) r|=(1ULL<<(i*8+ff)); for(int i=rr-1; i>=1; i--) r|=(1ULL<<(i*8+ff)); for(int i=ff+1; i<=6; i++) r|=(1ULL<<(rr*8+i)); for(int i=ff-1; i>=1; i--) r|=(1ULL<<(rr*8+i)); return r; }
static uint64_t calc_bishop_mask(int sq) { uint64_t r=0; int rr=sq/8, ff=sq%8; for(int i=rr+1,j=ff+1; i<=6&&j<=6; i++,j++) r|=(1ULL<<(i*8+j)); for(int i=rr+1,j=ff-1; i<=6&&j>=1; i++,j--) r|=(1ULL<<(i*8+j)); for(int i=rr-1,j=ff+1; i>=1&&j<=6; i--,j++) r|=(1ULL<<(i*8+j)); for(int i=rr-1,j=ff-1; i>=1&&j>=1; i--,j--) r|=(1ULL<<(i*8+j)); return r; }
static uint64_t calc_rook_attack(int sq, uint64_t block) { uint64_t r=0; int rr=sq/8, ff=sq%8; for(int i=rr+1; i<=7; i++) { r|=(1ULL<<(i*8+ff)); if(block&(1ULL<<(i*8+ff))) break; } for(int i=rr-1; i>=0; i--) { r|=(1ULL<<(i*8+ff)); if(block&(1ULL<<(i*8+ff))) break; } for(int i=ff+1; i<=7; i++) { r|=(1ULL<<(rr*8+i)); if(block&(1ULL<<(rr*8+i))) break; } for(int i=ff-1; i>=0; i--) { r|=(1ULL<<(rr*8+i)); if(block&(1ULL<<(rr*8+i))) break; } return r; }
static uint64_t calc_bishop_attack(int sq, uint64_t block) { uint64_t r=0; int rr=sq/8, ff=sq%8; for(int i=rr+1,j=ff+1; i<=7&&j<=7; i++,j++) { r|=(1ULL<<(i*8+j)); if(block&(1ULL<<(i*8+j))) break; } for(int i=rr+1,j=ff-1; i<=7&&j>=0; i++,j--) { r|=(1ULL<<(i*8+j)); if(block&(1ULL<<(i*8+j))) break; } for(int i=rr-1,j=ff+1; i>=0&&j<=7; i--,j++) { r|=(1ULL<<(i*8+j)); if(block&(1ULL<<(i*8+j))) break; } for(int i=rr-1,j=ff-1; i>=0&&j>=0; i--,j--) { r|=(1ULL<<(i*8+j)); if(block&(1ULL<<(i*8+j))) break; } return r; }
static int ctz(uint64_t bb) { int count=0; while((bb&1)==0){count++; bb>>=1;} return count; }
static uint64_t set_occ(int index, int bits, uint64_t mask) { uint64_t occ=0; for(int i=0; i<bits; i++) { int sq=ctz(mask); mask&=mask-1; if(index&(1<<i)) occ|=(1ULL<<sq); } return occ; }
static int popc(uint64_t bb) { int c=0; while(bb){c++; bb&=bb-1;} return c; }
void init_magics(void) {
    int rook_offset = 0, bishop_offset = 0;
    for(int sq=0; sq<64; sq++) {
        BISHOP_MASKS[sq] = calc_bishop_mask(sq);
        ROOK_MASKS[sq] = calc_rook_mask(sq);
        BISHOP_SHIFTS[sq] = 64 - popc(BISHOP_MASKS[sq]);
        ROOK_SHIFTS[sq] = 64 - popc(ROOK_MASKS[sq]);
        BISHOP_ATTACKS[sq] = &bishop_table[bishop_offset];
        ROOK_ATTACKS[sq] = &rook_table[rook_offset];
        int b_bits = popc(BISHOP_MASKS[sq]);
        for(int i=0; i<(1<<b_bits); i++) {
            uint64_t occ = set_occ(i, b_bits, BISHOP_MASKS[sq]);
            int magic_idx = (occ * BISHOP_MAGICS[sq]) >> BISHOP_SHIFTS[sq];
            BISHOP_ATTACKS[sq][magic_idx] = calc_bishop_attack(sq, occ);
        }
        bishop_offset += (1 << b_bits);
        int r_bits = popc(ROOK_MASKS[sq]);
        for(int i=0; i<(1<<r_bits); i++) {
            uint64_t occ = set_occ(i, r_bits, ROOK_MASKS[sq]);
            int magic_idx = (occ * ROOK_MAGICS[sq]) >> ROOK_SHIFTS[sq];
            ROOK_ATTACKS[sq][magic_idx] = calc_rook_attack(sq, occ);
        }
        rook_offset += (1 << r_bits);
    }
}
uint64_t rook_attacks(int sq, uint64_t occupied) { return ROOK_ATTACKS[sq][((occupied & ROOK_MASKS[sq]) * ROOK_MAGICS[sq]) >> ROOK_SHIFTS[sq]]; }
uint64_t bishop_attacks(int sq, uint64_t occupied) { return BISHOP_ATTACKS[sq][((occupied & BISHOP_MASKS[sq]) * BISHOP_MAGICS[sq]) >> BISHOP_SHIFTS[sq]]; }
uint64_t queen_attacks(int sq, uint64_t occupied) { return rook_attacks(sq, occupied) | bishop_attacks(sq, occupied); }
