// mlir_noise_ops.inc — MLIR emission for noise/error operations
//
// Noise sites and channels are inlined as constants at emit time.
// Each OP_NOISE unconditionally processes its site (no scheduling).

case Opcode::OP_NOISE: {
    uint32_t site_idx = ins.a;
    if (site_idx < flat.noise_sites.size()) {
        const auto& site = flat.noise_sites[site_idx];
        out << "  // noise site " << site_idx << " (scheduled)\n";
        // Check if next_noise_idx == site_idx
        std::string nni_cur = fresh_ssa();
        out << "  " << nni_cur << " = llvm.load %nni_ptr : !llvm.ptr -> i32\n";
        std::string site_val = emit_const_i32(out, site_idx);
        std::string nni_match = fresh_ssa();
        out << "  " << nni_match << " = llvm.icmp \"eq\" " << nni_cur << ", " << site_val << " : i32\n";
        std::string lbl_do_noise = fresh_label("do_noise");
        std::string lbl_skip_noise = fresh_label("skip_noise");
        out << "  llvm.cond_br " << nni_match << ", ^" << lbl_do_noise << ", ^" << lbl_skip_noise << "\n";
        out << "^" << lbl_do_noise << ":\n";
        // Roll RNG
        std::string u = emit_rng_uniform(out);
        std::string prob_sum_val = fresh_ssa();
        char psbuf[64]; snprintf(psbuf, sizeof(psbuf), "%.17e", site.prob_sum);
        out << "  " << prob_sum_val << " = llvm.mlir.constant(" << psbuf << " : f64) : f64\n";
        std::string roll = fresh_ssa();
        out << "  " << roll << " = llvm.fmul " << u << ", " << prob_sum_val << " : f64\n";
        // Check each channel: accumulate probability, apply Pauli when roll < cumulative
        std::string cumulative = fresh_ssa();
        out << "  " << cumulative << " = llvm.mlir.constant(0.0 : f64) : f64\n";
        for (uint32_t k = 0; k < site.count; ++k) {
            uint32_t ch_idx = site.offset + k;
            if (ch_idx >= flat.noise_channels.size()) break;
            const auto& ch = flat.noise_channels[ch_idx];
            // Add channel probability to cumulative
            std::string ch_prob = fresh_ssa();
            char cpbuf[64]; snprintf(cpbuf, sizeof(cpbuf), "%.17e", ch.prob);
            out << "  " << ch_prob << " = llvm.mlir.constant(" << cpbuf << " : f64) : f64\n";
            std::string new_cum = fresh_ssa();
            out << "  " << new_cum << " = llvm.fadd " << cumulative << ", " << ch_prob << " : f64\n";
            // Check if roll < new_cumulative (and roll >= old cumulative)
            std::string hit = fresh_ssa();
            out << "  " << hit << " = llvm.fcmp \"olt\" " << roll << ", " << new_cum << " : f64\n";
            std::string not_yet = fresh_ssa();
            out << "  " << not_yet << " = llvm.fcmp \"oge\" " << roll << ", " << cumulative << " : f64\n";
            std::string apply = fresh_ssa();
            out << "  " << apply << " = llvm.and " << hit << ", " << not_yet << " : i1\n";
            // If apply: XOR ch.x into px, ch.z into pz (apply_pauli_to_frame)
            std::string lbl_apply = fresh_label("nch_apply");
            std::string lbl_skip = fresh_label("nch_skip");
            out << "  llvm.cond_br " << apply << ", ^" << lbl_apply << ", ^" << lbl_skip << "\n";
            out << "^" << lbl_apply << ":\n";
            // XOR ch.x[0] into px[0], ch.x[1] into px[1]
            for (int w = 0; w < 2; ++w) {
                if (ch.x[w] != 0) {
                    char xbuf[32]; snprintf(xbuf, sizeof(xbuf), "%llu", (unsigned long long)ch.x[w]);
                    std::string xval = fresh_ssa();
                    out << "  " << xval << " = llvm.mlir.constant(" << xbuf << " : i64) : i64\n";
                    std::string pw = fresh_ssa();
                    std::string widx = fresh_ssa();
                    out << "  " << widx << " = llvm.mlir.constant(" << w << " : i64) : i64\n";
                    out << "  " << pw << " = llvm.getelementptr inbounds %px_ptr["
                        << widx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
                    std::string old_val = fresh_ssa();
                    out << "  " << old_val << " = llvm.load " << pw << " : !llvm.ptr -> i64\n";
                    std::string new_val = fresh_ssa();
                    out << "  " << new_val << " = llvm.xor " << old_val << ", " << xval << " : i64\n";
                    out << "  llvm.store " << new_val << ", " << pw << " : i64, !llvm.ptr\n";
                }
            }
            // XOR ch.z[0] into pz[0], ch.z[1] into pz[1]
            for (int w = 0; w < 2; ++w) {
                if (ch.z[w] != 0) {
                    char zbuf[32]; snprintf(zbuf, sizeof(zbuf), "%llu", (unsigned long long)ch.z[w]);
                    std::string zval = fresh_ssa();
                    out << "  " << zval << " = llvm.mlir.constant(" << zbuf << " : i64) : i64\n";
                    std::string pw = fresh_ssa();
                    std::string widx = fresh_ssa();
                    out << "  " << widx << " = llvm.mlir.constant(" << w << " : i64) : i64\n";
                    out << "  " << pw << " = llvm.getelementptr inbounds %pz_ptr["
                        << widx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
                    std::string old_val = fresh_ssa();
                    out << "  " << old_val << " = llvm.load " << pw << " : !llvm.ptr -> i64\n";
                    std::string new_val = fresh_ssa();
                    out << "  " << new_val << " = llvm.xor " << old_val << ", " << zval << " : i64\n";
                    out << "  llvm.store " << new_val << ", " << pw << " : i64, !llvm.ptr\n";
                }
            }
            out << "  llvm.br ^" << lbl_skip << "\n";
            out << "^" << lbl_skip << ":\n";
            cumulative = new_cum;
        }
        // Update next_noise_idx and draw next
        std::string next_si = emit_const_i32(out, site_idx + 1);
        out << "  llvm.store " << next_si << ", %nni_ptr : i32, !llvm.ptr\n";
        emit_draw_next_noise(out, flat);
        out << "  llvm.br ^" << lbl_skip_noise << "\n";
        out << "^" << lbl_skip_noise << ":\n";
    }
    break;
}
case Opcode::OP_NOISE_BLOCK: {
    // Process noise sites in [ins.a, ins.a + ins.b) with scheduling
    uint32_t start = ins.a;
    uint32_t end = start + ins.b;
    for (uint32_t si = start; si < end && si < flat.noise_sites.size(); ++si) {
        const auto& site = flat.noise_sites[si];
        out << "  // noise_block site " << si << " (scheduled)\n";
        // Check scheduling
        std::string nni_cur = fresh_ssa();
        out << "  " << nni_cur << " = llvm.load %nni_ptr : !llvm.ptr -> i32\n";
        std::string site_val = emit_const_i32(out, si);
        std::string nni_match = fresh_ssa();
        out << "  " << nni_match << " = llvm.icmp \"eq\" " << nni_cur << ", " << site_val << " : i32\n";
        std::string lbl_do = fresh_label("nb_do");
        std::string lbl_end = fresh_label("nb_end");
        out << "  llvm.cond_br " << nni_match << ", ^" << lbl_do << ", ^" << lbl_end << "\n";
        out << "^" << lbl_do << ":\n";
        std::string u = emit_rng_uniform(out);
        std::string prob_sum_val = fresh_ssa();
        char psbuf[64]; snprintf(psbuf, sizeof(psbuf), "%.17e", site.prob_sum);
        out << "  " << prob_sum_val << " = llvm.mlir.constant(" << psbuf << " : f64) : f64\n";
        std::string roll = fresh_ssa();
        out << "  " << roll << " = llvm.fmul " << u << ", " << prob_sum_val << " : f64\n";
        std::string cumulative = fresh_ssa();
        out << "  " << cumulative << " = llvm.mlir.constant(0.0 : f64) : f64\n";
        for (uint32_t k = 0; k < site.count; ++k) {
            uint32_t ch_idx = site.offset + k;
            if (ch_idx >= flat.noise_channels.size()) break;
            const auto& ch = flat.noise_channels[ch_idx];
            std::string ch_prob = fresh_ssa();
            char cpbuf[64]; snprintf(cpbuf, sizeof(cpbuf), "%.17e", ch.prob);
            out << "  " << ch_prob << " = llvm.mlir.constant(" << cpbuf << " : f64) : f64\n";
            std::string new_cum = fresh_ssa();
            out << "  " << new_cum << " = llvm.fadd " << cumulative << ", " << ch_prob << " : f64\n";
            std::string hit = fresh_ssa();
            out << "  " << hit << " = llvm.fcmp \"olt\" " << roll << ", " << new_cum << " : f64\n";
            std::string not_yet = fresh_ssa();
            out << "  " << not_yet << " = llvm.fcmp \"oge\" " << roll << ", " << cumulative << " : f64\n";
            std::string apply = fresh_ssa();
            out << "  " << apply << " = llvm.and " << hit << ", " << not_yet << " : i1\n";
            std::string lbl_apply = fresh_label("nb_apply");
            std::string lbl_skip = fresh_label("nb_skip");
            out << "  llvm.cond_br " << apply << ", ^" << lbl_apply << ", ^" << lbl_skip << "\n";
            out << "^" << lbl_apply << ":\n";
            for (int w = 0; w < 2; ++w) {
                if (ch.x[w] != 0) {
                    std::string xval = emit_const_i64(out, ch.x[w]);
                    std::string widx = emit_const_i64(out, w);
                    std::string pw = fresh_ssa();
                    out << "  " << pw << " = llvm.getelementptr inbounds %px_ptr["
                        << widx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
                    std::string ov = fresh_ssa();
                    out << "  " << ov << " = llvm.load " << pw << " : !llvm.ptr -> i64\n";
                    std::string nv = fresh_ssa();
                    out << "  " << nv << " = llvm.xor " << ov << ", " << xval << " : i64\n";
                    out << "  llvm.store " << nv << ", " << pw << " : i64, !llvm.ptr\n";
                }
            }
            for (int w = 0; w < 2; ++w) {
                if (ch.z[w] != 0) {
                    std::string zval = emit_const_i64(out, ch.z[w]);
                    std::string widx = emit_const_i64(out, w);
                    std::string pw = fresh_ssa();
                    out << "  " << pw << " = llvm.getelementptr inbounds %pz_ptr["
                        << widx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
                    std::string ov = fresh_ssa();
                    out << "  " << ov << " = llvm.load " << pw << " : !llvm.ptr -> i64\n";
                    std::string nv = fresh_ssa();
                    out << "  " << nv << " = llvm.xor " << ov << ", " << zval << " : i64\n";
                    out << "  llvm.store " << nv << ", " << pw << " : i64, !llvm.ptr\n";
                }
            }
            out << "  llvm.br ^" << lbl_skip << "\n";
            out << "^" << lbl_skip << ":\n";
            cumulative = new_cum;
        }
        // Advance next_noise_idx and draw
        std::string next_si = emit_const_i32(out, si + 1);
        out << "  llvm.store " << next_si << ", %nni_ptr : i32, !llvm.ptr\n";
        emit_draw_next_noise(out, flat);
        out << "  llvm.br ^" << lbl_end << "\n";
        out << "^" << lbl_end << ":\n";
    }
    break;
}
case Opcode::OP_READOUT_NOISE: {
    // Readout noise: flip measurement bit with probability
    if (ins.a < flat.readout_noise.size()) {
        const auto& rn = flat.readout_noise[ins.a];
        std::string u = emit_rng_uniform(out);
        std::string prob = fresh_ssa();
        char pbuf[64]; snprintf(pbuf, sizeof(pbuf), "%.17e", rn.prob);
        out << "  " << prob << " = llvm.mlir.constant(" << pbuf << " : f64) : f64\n";
        std::string flip = fresh_ssa();
        out << "  " << flip << " = llvm.fcmp \"olt\" " << u << ", " << prob << " : f64\n";
        std::string lbl_flip = fresh_label("rn_flip");
        std::string lbl_noflip = fresh_label("rn_noflip");
        out << "  llvm.cond_br " << flip << ", ^" << lbl_flip << ", ^" << lbl_noflip << "\n";
        out << "^" << lbl_flip << ":\n";
        std::string midx = emit_const_i64(out, rn.meas_idx);
        std::string mptr = fresh_ssa();
        out << "  " << mptr << " = llvm.getelementptr inbounds %meas_ptr["
            << midx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        std::string mval = fresh_ssa();
        out << "  " << mval << " = llvm.load " << mptr << " : !llvm.ptr -> i8\n";
        std::string flipped = fresh_ssa();
        out << "  " << flipped << " = llvm.xor " << mval << ", %c1_i8 : i8\n";
        out << "  llvm.store " << flipped << ", " << mptr << " : i8, !llvm.ptr\n";
        out << "  llvm.br ^" << lbl_noflip << "\n";
        out << "^" << lbl_noflip << ":\n";
    }
    break;
}
case Opcode::OP_POSTSELECT: {
    bool expected_one = (ins.flags & kFlagExpectedOne) != 0;
    if (ins.a + 1 < flat.detector_offsets.size()) {
        uint32_t ps_start = flat.detector_offsets[ins.a];
        uint32_t ps_end = flat.detector_offsets[ins.a + 1];
        std::string parity = fresh_ssa();
        out << "  " << parity << " = llvm.mlir.constant("
            << (expected_one ? 1 : 0) << " : i8) : i8\n";
        for (uint32_t k = ps_start; k < ps_end; ++k) {
            char tidx_buf[32]; snprintf(tidx_buf, sizeof(tidx_buf), "%u", flat.detector_targets[k]);
            std::string tidx = fresh_ssa();
            out << "  " << tidx << " = llvm.mlir.constant(" << tidx_buf << " : i64) : i64\n";
            std::string tptr = fresh_ssa();
            out << "  " << tptr << " = llvm.getelementptr inbounds %meas_ptr["
                << tidx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
            std::string tval = fresh_ssa();
            out << "  " << tval << " = llvm.load " << tptr << " : !llvm.ptr -> i8\n";
            std::string new_parity = fresh_ssa();
            out << "  " << new_parity << " = llvm.xor " << parity << ", " << tval << " : i8\n";
            parity = new_parity;
        }
        std::string ps_fail = fresh_ssa();
        out << "  " << ps_fail << " = llvm.icmp \"ne\" " << parity << ", %c0_i8 : i8\n";
        std::string lbl_ps_disc = fresh_label("ps_disc");
        std::string lbl_ps_ok = fresh_label("ps_ok");
        out << "  llvm.cond_br " << ps_fail << ", ^" << lbl_ps_disc << ", ^" << lbl_ps_ok << "\n";
        out << "^" << lbl_ps_disc << ":\n";
        out << "  llvm.store %c1_i8, %discarded_ptr : i8, !llvm.ptr\n";
        out << "  llvm.br ^" << lbl_ps_ok << "\n";
        out << "^" << lbl_ps_ok << ":\n";
    }
    break;
}
case Opcode::OP_APPLY_PAULI: {
    // Inline pauli mask application
    if (ins.a < flat.pauli_masks.size()) {
        const auto& mask = flat.pauli_masks[ins.a];
        for (int w = 0; w < 2; ++w) {
            if (mask.x[w] != 0) {
                std::string xval = emit_const_i64(out, mask.x[w]);
                std::string widx = emit_const_i64(out, w);
                std::string pw = fresh_ssa();
                out << "  " << pw << " = llvm.getelementptr inbounds %px_ptr["
                    << widx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
                std::string ov = fresh_ssa();
                out << "  " << ov << " = llvm.load " << pw << " : !llvm.ptr -> i64\n";
                std::string nv = fresh_ssa();
                out << "  " << nv << " = llvm.xor " << ov << ", " << xval << " : i64\n";
                out << "  llvm.store " << nv << ", " << pw << " : i64, !llvm.ptr\n";
            }
            if (mask.z[w] != 0) {
                std::string zval = emit_const_i64(out, mask.z[w]);
                std::string widx = emit_const_i64(out, w);
                std::string pw = fresh_ssa();
                out << "  " << pw << " = llvm.getelementptr inbounds %pz_ptr["
                    << widx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i64\n";
                std::string ov = fresh_ssa();
                out << "  " << ov << " = llvm.load " << pw << " : !llvm.ptr -> i64\n";
                std::string nv = fresh_ssa();
                out << "  " << nv << " = llvm.xor " << ov << ", " << zval << " : i64\n";
                out << "  llvm.store " << nv << ", " << pw << " : i64, !llvm.ptr\n";
            }
        }
    }
    break;
}
