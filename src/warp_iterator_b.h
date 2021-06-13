#pragma once

#include "utils.h"

class WarpIteratorB {
public:
  CUTLASS_HOST_DEVICE
  WarpIteratorB(float* data) {
    int32_t shift = threadIdx.x / 128;
    int32_t residual = threadIdx.x % 16;
    int32_t column = residual / 2;
    ptr_ = (Array<float, 4>*)data + shift * 16 + column;
  }

  CUTLASS_HOST_DEVICE
  void reset() {
    ptr_ -= 2 * 128;
  }

  CUTLASS_HOST_DEVICE
  WarpIteratorB & operator++() {
    ptr_ += 32;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  void load(Array<float, 8> &frag) const {
    Array<float, 4> *dst_ptr = reinterpret_cast<Array<float, 4> *>(&frag);
    dst_ptr[0] = ptr_[0];
    dst_ptr[1] = ptr_[8];
  }
private:
  Array<float, 4>* ptr_;
};

// 000 002 004 006 008 010 012 014  000 002 004 006 008 010 012 014  128 130 132 134 136 138 140 142  128 130 132 134 136 138 140 142
// 001 003 005 007 009 011 013 015  001 003 005 007 009 011 013 015  129 131 133 135 137 139 141 143  129 131 133 135 137 139 141 143
// 016 018 020 022 024 026 028 030  016 018 020 022 024 026 028 030  144 146 148 150 152 154 156 158  144 146 148 150 152 154 156 158
// 017 019 021 023 025 027 029 031  017 019 021 023 025 027 029 031  145 147 149 151 153 155 157 159  145 147 149 151 153 155 157 159

// 000 002 004 006 008 010 012 014  000 002 004 006 008 010 012 014  128 130 132 134 136 138 140 142  128 130 132 134 136 138 140 142
// 001 003 005 007 009 011 013 015  001 003 005 007 009 011 013 015  129 131 133 135 137 139 141 143  129 131 133 135 137 139 141 143
// 016 018 020 022 024 026 028 030  016 018 020 022 024 026 028 030  144 146 148 150 152 154 156 158  144 146 148 150 152 154 156 158
// 017 019 021 023 025 027 029 031  017 019 021 023 025 027 029 031  145 147 149 151 153 155 157 159  145 147 149 151 153 155 157 159

// 032 034 036 038 040 042 044 046  032 034 036 038 040 042 044 046  160 162 164 166 168 170 172 174  160 162 164 166 168 170 172 174
// 033 035 037 039 041 043 045 047  033 035 037 039 041 043 045 047  161 163 165 167 169 171 173 175  161 163 165 167 169 171 173 175
// 048 050 052 054 056 058 060 062  048 050 052 054 056 058 060 062  176 178 180 182 184 186 188 190  176 178 180 182 184 186 188 190
// 049 051 053 055 057 059 061 063  049 051 053 055 057 059 061 063  177 179 181 183 185 187 189 191  177 179 181 183 185 187 189 191

// 032 034 036 038 040 042 044 046  032 034 036 038 040 042 044 046  160 162 164 166 168 170 172 174  160 162 164 166 168 170 172 174
// 033 035 037 039 041 043 045 047  033 035 037 039 041 043 045 047  161 163 165 167 169 171 173 175  161 163 165 167 169 171 173 175
// 048 050 052 054 056 058 060 062  048 050 052 054 056 058 060 062  176 178 180 182 184 186 188 190  176 178 180 182 184 186 188 190
// 049 051 053 055 057 059 061 063  049 051 053 055 057 059 061 063  177 179 181 183 185 187 189 191  177 179 181 183 185 187 189 191

// 064 066 068 070 072 074 076 078  064 066 068 070 072 074 076 078  192 194 196 198 200 202 204 206  192 194 196 198 200 202 204 206
// 065 067 069 071 073 075 077 079  065 067 069 071 073 075 077 079  193 195 197 199 201 203 205 207  193 195 197 199 201 203 205 207
// 080 082 084 086 088 090 092 094  080 082 084 086 088 090 092 094  208 210 212 214 216 218 220 222  208 210 212 214 216 218 220 222
// 081 083 085 087 089 091 093 095  081 083 085 087 089 091 093 095  209 211 213 215 217 219 221 223  209 211 213 215 217 219 221 223

// 064 066 068 070 072 074 076 078  064 066 068 070 072 074 076 078  192 194 196 198 200 202 204 206  192 194 196 198 200 202 204 206
// 065 067 069 071 073 075 077 079  065 067 069 071 073 075 077 079  193 195 197 199 201 203 205 207  193 195 197 199 201 203 205 207
// 080 082 084 086 088 090 092 094  080 082 084 086 088 090 092 094  208 210 212 214 216 218 220 222  208 210 212 214 216 218 220 222
// 081 083 085 087 089 091 093 095  081 083 085 087 089 091 093 095  209 211 213 215 217 219 221 223  209 211 213 215 217 219 221 223

// 096 098 100 102 104 106 108 110  096 098 100 102 104 106 108 110  224 226 228 230 232 234 236 238  224 226 228 230 232 234 236 238
// 097 099 101 103 105 107 109 111  097 099 101 103 105 107 109 111  225 227 229 231 233 235 237 239  225 227 229 231 233 235 237 239
// 112 114 116 118 120 122 124 126  112 114 116 118 120 122 124 126  240 242 244 246 248 250 252 254  240 242 244 246 248 250 252 254
// 113 115 117 119 121 123 125 127  113 115 117 119 121 123 125 127  241 243 245 247 249 251 253 255  241 243 245 247 249 251 253 255

// 096 098 100 102 104 106 108 110  096 098 100 102 104 106 108 110  224 226 228 230 232 234 236 238  224 226 228 230 232 234 236 238
// 097 099 101 103 105 107 109 111  097 099 101 103 105 107 109 111  225 227 229 231 233 235 237 239  225 227 229 231 233 235 237 239
// 112 114 116 118 120 122 124 126  112 114 116 118 120 122 124 126  240 242 244 246 248 250 252 254  240 242 244 246 248 250 252 254
// 113 115 117 119 121 123 125 127  113 115 117 119 121 123 125 127  241 243 245 247 249 251 253 255  241 243 245 247 249 251 253 255