// Optional single-threaded Strata BS=1 replay; no fast-math, float32 scores.
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

namespace fs = std::filesystem;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void number(std::string& s, int64_t value) {
    char buffer[32]; auto end=std::to_chars(buffer,buffer+32,value).ptr; s.append(buffer,end);
}
struct Events {
    gzFile file; uint64_t id, tick, transfer; uint64_t demands=0, swaps=0;
    Events(const fs::path& path, uint64_t a, uint64_t b, uint64_t c):id(a),tick(b),transfer(c) {
        file=gzopen(path.c_str(),"ab1"); require(file,"cannot open events"); gzbuffer(file,1<<20);
    }
    void event(const char* kind, int64_t step, int layer, int expert, bool adapt, bool evict=false) {
        std::string s; s.reserve(400);
        s+="{\"event_id\":"; number(s,++id); s+=",\"phase\":\"evaluation\",\"step_id\":"; number(s,step);
        s+=adapt?",\"hook\":\"on_step_end\",\"kind\":\"":",\"hook\":\"on_routes_ready\",\"kind\":\"";
        s+=kind; s+="\",\"layer_id\":"; number(s,layer); s+=",\"expert_id\":"; number(s,expert);
        s+=",\"units\":1,\"reason\":\""; s+=adapt?"adaptation":"demand"; s+="\",\"transfer_id\":";
        if(evict) s+="null"; else number(s,transfer);
        s+=",\"destination\":";
        if(evict) s+="null"; else s+=adapt?"\"cache\"":"\"workspace\"";
        s+=",\"context_layer_id\":"; if(adapt) s+="null"; else number(s,layer);
        s+=",\"logical_tick\":"; number(s,++tick); s+=",\"simulated_time\":null}\n";
        require(gzwrite(file,s.data(),unsigned(s.size()))==int(s.size()),"event write failed");
    }
    void load(int64_t step, int layer, int expert, bool adapt) {
        ++transfer; if(adapt) ++swaps; else ++demands;
        event("transfer_start",step,layer,expert,adapt);
        event("transfer_complete",step,layer,expert,adapt);
    }
    ~Events() { if(file) gzclose(file); }
};
struct Access {
    fs::path root; int topk, limit, rows=0, shard=0; std::vector<char> data;
    std::vector<std::pair<std::string,int>> shards;
    Access(fs::path r,int k,int n):root(r),topk(k),limit(n) { data.reserve(size_t(n)*(32+2*k)); }
    void u32(uint32_t x) { for(int i=0;i<4;++i)data.push_back(char(x>>(8*i))); }
    void u64(uint64_t x) { for(int i=0;i<8;++i)data.push_back(char(x>>(8*i))); }
    void append(uint64_t tick,uint64_t step,int request,int token,int layer,const uint8_t* experts,const uint8_t* hits) {
        u64(tick); u64(step); u32(request); u32(token); u32(layer); int count=0;
        for(int i=0;i<topk;++i)data.push_back(char(experts[i]));
        for(int i=0;i<topk;++i){data.push_back(char(hits[i]));count+=hits[i];}
        u32(count); if(++rows==limit)flush();
    }
    void flush() {
        if(!rows)return;
        std::string header="{'descr': [('probe_tick', '<u8'), ('step_id', '<u8'), ('request_index', '<u4'), ('token_index', '<u4'), ('layer_id', '<u4'), ('expert_ids', '|u1', ("+std::to_string(topk)+",)), ('hit_mask', '|b1', ("+std::to_string(topk)+",)), ('hit_count', '<u4')], 'fortran_order': False, 'shape': ("+std::to_string(rows)+",), }";
        size_t padded=((10+header.size()+1+63)/64)*64-10;
        header.resize(padded-1,' '); header+='\n'; require(header.size()<65536,"header too long");
        char name[64]; std::snprintf(name,sizeof(name),"access-%06d.npy",shard++);
        std::ofstream f(root/name,std::ios::binary); f.write("\x93NUMPY\x01\x00",8);
        char length[2]={char(header.size()&255),char(header.size()>>8)};
        f.write(length,2); f.write(header.data(),header.size()); f.write(data.data(),data.size());
        require(bool(f),"access write failed"); shards.push_back({name,rows}); data.clear();rows=0;
    }
};
struct Swap { double gain; int incoming, outgoing; };

// Independently scan the sealed event stream, checking its counters and load pairs.
void audit(const fs::path& path) {
    gzFile f=gzopen(path.c_str(),"rb"); require(f,"cannot open audit events"); gzbuffer(f,1<<20);
    char line[2048]; uint64_t events=0,last_transfer=0,pending=0;
    std::array<uint64_t,3> loads{},units{},evictions{}; uint64_t initialization=0,demand=0,adaptation=0;
    auto value=[&](const char* key)->uint64_t {
        auto start=std::strstr(line,key);require(start,"missing numeric field");start+=std::strlen(key);
        uint64_t v=0;auto parsed=std::from_chars(start,start+std::strlen(start),v);require(parsed.ec==std::errc(),"bad numeric field");return v;
    };
    while(gzgets(f,line,sizeof(line))) {
        require(std::strchr(line,'\n'),"event line too long"); ++events;
        require(value("\"event_id\":")==events && value("\"logical_tick\":")==events,"event sequence mismatch");
        int phase=std::strstr(line,"\"phase\":\"initialization\"")?0:std::strstr(line,"\"phase\":\"evaluation\"")?2:1;
        auto size=value("\"units\":");
        if(std::strstr(line,"\"kind\":\"transfer_start\"")) {
            require(!pending,"overlapping instantaneous transfers"); pending=value("\"transfer_id\":");require(pending==++last_transfer,"transfer sequence mismatch");
        } else if(std::strstr(line,"\"kind\":\"transfer_complete\"")) {
            require(pending && pending==value("\"transfer_id\":"),"unmatched transfer");pending=0;
            ++loads[phase];units[phase]+=size;
            if(std::strstr(line,"\"reason\":\"initialization\""))initialization+=size;
            else if(std::strstr(line,"\"reason\":\"demand\""))demand+=size;
            else if(std::strstr(line,"\"reason\":\"adaptation\""))adaptation+=size;
            else throw std::runtime_error("unexpected transfer reason");
        } else { require(std::strstr(line,"\"kind\":\"evict\""),"unexpected event kind");require(!pending,"evict during transfer");++evictions[phase]; }
    }
    require(gzeof(f) && !pending,"incomplete event stream"); require(gzclose(f)==Z_OK,"audit close failed");
    std::cout<<"{\"event_count\":"<<events<<",\"initialization_units\":"<<initialization<<",\"demand_units\":"<<demand<<",\"adaptation_units\":"<<adaptation<<",\"phases\":[";
    for(int i=0;i<3;++i){if(i)std::cout<<',';std::cout<<"{\"load_count\":"<<loads[i]<<",\"units\":"<<units[i]<<",\"evictions\":"<<evictions[i]<<'}';}
    std::cout<<"]}\n";
}

int main(int argc,char** argv) { try {
    if(argc==3 && std::string(argv[1])=="--audit"){audit(argv[2]);return 0;}
    require(argc==4,"usage: executable settings input_routes detail_directory");
    std::ifstream settings(argv[1]);int requests,layers,experts,topk,adapt_every,max_swaps,chunk_rows;double min_count,min_gain;float decay;
    settings>>requests>>layers>>experts>>topk>>adapt_every>>max_swaps>>min_count>>min_gain>>decay>>chunk_rows;
    require(requests>0 && layers>0 && experts<=256 && topk>0 && topk<=experts,"invalid dimensions");
    std::vector<int> layer_ids(layers),lengths(requests);for(auto& x:layer_ids)settings>>x;for(auto& x:lengths)settings>>x;
    int capacity;settings>>capacity;std::vector<uint8_t> resident(layers*experts),seen(layers*experts);std::vector<float> usage(layers*experts);
    for(int i=0;i<capacity;++i){int key;settings>>key;require(key>=0&&key<int(resident.size())&&!resident[key],"invalid initial residency");resident[key]=1;}
    uint64_t event_id,tick,transfer;settings>>event_id>>tick>>transfer;require(bool(settings),"invalid settings");
    std::ifstream input(argv[2],std::ios::binary);require(bool(input),"cannot open routes");
    Events events(fs::path(argv[3])/"cache_events.jsonl.gz",event_id,tick,transfer);
    Access access(argv[3],topk,chunk_rows);std::vector<uint8_t> batch(layers*topk),mask(topk);std::vector<int> seen_order;
    std::vector<int> incoming,outgoing;std::vector<Swap> swaps;uint64_t step=0,adaptations=0;
    for(int r=0;r<requests;++r) {
        for(int token=0;token<lengths[r];++token,++step) {
            input.read(reinterpret_cast<char*>(batch.data()),batch.size());require(bool(input),"truncated route input");
            for(int l=0;l<layers;++l) {
                auto routes=batch.data()+l*topk;uint64_t probe_tick=events.tick;
                for(int k=0;k<topk;++k){int key=l*experts+routes[k];require(routes[k]<experts,"invalid expert ID");mask[k]=resident[key];if(!seen[key]){seen[key]=1;seen_order.push_back(key);}usage[key]=usage[key]+1.0f;}
                for(int k=0;k<topk;++k)if(!mask[k])events.load(step,layer_ids[l],routes[k],false);
                access.append(probe_tick,step,r,token,layer_ids[l],routes,mask.data());
            }
            if(adapt_every && max_swaps && (step+1)%adapt_every==0) {
                swaps.clear();
                for(int l=0;l<layers;++l) {
                    incoming.clear();outgoing.clear();
                    for(int e=0;e<experts;++e){int key=l*experts+e;if(resident[key])outgoing.push_back(key);else if(seen[key]&&usage[key]>=min_count)incoming.push_back(key);}
                    std::sort(incoming.begin(),incoming.end(),[&](int a,int b){return usage[a]!=usage[b]?usage[a]>usage[b]:a<b;});
                    std::sort(outgoing.begin(),outgoing.end(),[&](int a,int b){return usage[a]!=usage[b]?usage[a]<usage[b]:a<b;});
                    for(size_t i=0;i<std::min(incoming.size(),outgoing.size());++i){int a=incoming[i],b=outgoing[i];double hot=usage[a],cold=usage[b];if(hot<cold+min_gain)break;swaps.push_back({hot-cold,a,b});}
                }
                std::sort(swaps.begin(),swaps.end(),[](const Swap& a,const Swap& b){if(a.gain!=b.gain)return a.gain>b.gain;if(a.incoming!=b.incoming)return a.incoming<b.incoming;return a.outgoing<b.outgoing;});
                // Python builds the actions, then decays every observed float32 score.
                for(int key:seen_order)usage[key]=usage[key]*decay;
                for(size_t i=0;i<std::min(swaps.size(),size_t(max_swaps));++i){auto s=swaps[i];resident[s.outgoing]=0;events.event("evict",step,layer_ids[s.outgoing/experts],s.outgoing%experts,true,true);resident[s.incoming]=1;events.load(step,layer_ids[s.incoming/experts],s.incoming%experts,true);}
                ++adaptations;
            }
        }
        if((r+1)%20==0 || r+1==requests)std::cout<<"PROGRESS requests="<<r+1<<'/'<<requests<<" steps="<<step<<" demand="<<events.demands<<" swaps="<<events.swaps<<std::endl;
    }
    require(input.peek()==std::char_traits<char>::eof(),"extra route input");access.flush();
    std::ofstream result(fs::path(argv[3]).parent_path()/"native-result.json");result<<std::setprecision(std::numeric_limits<float>::max_digits10);
    result<<"{\"steps\":"<<step<<",\"adaptations\":"<<adaptations<<",\"swaps\":"<<events.swaps<<",\"demand_loads\":"<<events.demands<<",\"event_id\":"<<events.id<<",\"tick\":"<<events.tick<<",\"transfer_id\":"<<events.transfer<<",\"resident\":[";
    bool first=true;for(int key=0;key<int(resident.size());++key)if(resident[key]){if(!first)result<<',';first=false;result<<'['<<layer_ids[key/experts]<<','<<key%experts<<']';}
    result<<"],\"usage\":[";first=true;for(int key:seen_order){if(!first)result<<',';first=false;result<<'['<<layer_ids[key/experts]<<','<<key%experts<<','<<usage[key]<<']';}
    result<<"],\"shards\":[";first=true;for(auto& s:access.shards){if(!first)result<<',';first=false;result<<"{\"path\":\""<<s.first<<"\",\"rows\":"<<s.second<<'}';}result<<"]}\n";require(bool(result),"result write failed");
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;} }
