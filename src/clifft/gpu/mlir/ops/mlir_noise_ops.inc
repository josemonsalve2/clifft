// mlir_noise_ops.inc — MLIR emission for noise/error operations
//
// Noise data is loaded from kernel pointer arguments at runtime.
// Layout: noise_sites_ptr → GpuNoiseSite[n] (16 bytes each: u32 offset, u32 count, f64 prob_sum)
//         noise_channels_ptr → GpuChannel[m] (40 bytes each: u64 x[2], u64 z[2], f64 prob)

case Opcode::OP_NOISE: {
    // COOP: run the whole site body on tid0 only (matches gold
    // hip_sampler.hip:1677). All coop scalar RNG draws are tid0-only, so tid0's
    // per-thread nni + RNG stream stays aligned with SVM; the other 255 threads
    // previously did identical redundant work (RNG + channel loop) — pure waste.
    // The trailing barrier publishes tid0's frame writes to all threads.
    uint32_t site_idx = ins.a;
    out << "  // noise site " << site_idx << " (runtime data)\n";
    std::string noise_t0 = cooperative_mode ? emit_tid0_guard_begin(out) : "";
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
    // Load site metadata from noise_sites_ptr (GpuNoiseSite = 16 bytes)
    // byte offset = site_idx * 16; fields: offset@0, count@4, prob_sum@8
    std::string site_byte_off = emit_const_i64(out, (uint64_t)site_idx * 16);
    std::string site_base = fresh_ssa();
    out << "  " << site_base << " = llvm.getelementptr %noise_sites_ptr["
        << site_byte_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
    // Load offset (u32 at byte 0)
    std::string ch_offset = fresh_ssa();
    out << "  " << ch_offset << " = llvm.load " << site_base << " : !llvm.ptr -> i32\n";
    // Load count (u32 at byte 4)
    std::string count_off = emit_const_i64(out, 4);
    std::string count_ptr = fresh_ssa();
    out << "  " << count_ptr << " = llvm.getelementptr " << site_base << "["
        << count_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
    std::string ch_count = fresh_ssa();
    out << "  " << ch_count << " = llvm.load " << count_ptr << " : !llvm.ptr -> i32\n";
    // Load prob_sum (f64 at byte 8)
    std::string ps_off = emit_const_i64(out, 8);
    std::string ps_ptr = fresh_ssa();
    out << "  " << ps_ptr << " = llvm.getelementptr " << site_base << "["
        << ps_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
    std::string prob_sum_val = fresh_ssa();
    out << "  " << prob_sum_val << " = llvm.load " << ps_ptr << " : !llvm.ptr -> f64\n";
    // Roll RNG
    std::string u = emit_rng_uniform(out);
    std::string roll = fresh_ssa();
    out << "  " << roll << " = llvm.fmul " << u << ", " << prob_sum_val << " : f64\n";
    // Channel loop: iterate [0, ch_count), accumulate probability, apply Pauli
    std::string ch_count_64 = fresh_ssa();
    out << "  " << ch_count_64 << " = llvm.zext " << ch_count << " : i32 to i64\n";
    std::string ch_offset_64 = fresh_ssa();
    out << "  " << ch_offset_64 << " = llvm.zext " << ch_offset << " : i32 to i64\n";
    std::string fzero = fresh_ssa();
    out << "  " << fzero << " = llvm.mlir.constant(0.0 : f64) : f64\n";
    std::string nch_var = fresh_ssa(), nch_cum_var = fresh_ssa();
    std::string nch_hdr = fresh_label("nch_hdr"), nch_body = fresh_label("nch_body"), nch_done = fresh_label("nch_done");
    out << "  llvm.br ^" << nch_hdr << "(%c0_i64, " << fzero << " : i64, f64)\n";
    out << "^" << nch_hdr << "(" << nch_var << ": i64, " << nch_cum_var << ": f64):\n";
    std::string nch_cond = fresh_ssa();
    out << "  " << nch_cond << " = llvm.icmp \"ult\" " << nch_var << ", " << ch_count_64 << " : i64\n";
    out << "  llvm.cond_br " << nch_cond << ", ^" << nch_body << ", ^" << nch_done << "\n";
    out << "^" << nch_body << ":\n";
    // Channel address: noise_channels_ptr + (ch_offset + k) * 40
    std::string global_ch_idx = fresh_ssa();
    out << "  " << global_ch_idx << " = llvm.add " << ch_offset_64 << ", " << nch_var << " : i64\n";
    std::string ch_stride = emit_const_i64(out, sizeof(GpuChannel));
    std::string ch_byte_off = fresh_ssa();
    out << "  " << ch_byte_off << " = llvm.mul " << global_ch_idx << ", " << ch_stride << " : i64\n";
    std::string ch_base = fresh_ssa();
    out << "  " << ch_base << " = llvm.getelementptr %noise_channels_ptr["
        << ch_byte_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
    // Load ch.prob (f64) — offset depends on kPauliWords (x[PW], z[PW], prob).
    std::string prob_off = emit_const_i64(out, offsetof(GpuChannel, prob));
    std::string prob_ptr = fresh_ssa();
    out << "  " << prob_ptr << " = llvm.getelementptr " << ch_base << "["
        << prob_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
    std::string ch_prob = fresh_ssa();
    out << "  " << ch_prob << " = llvm.load " << prob_ptr << " : !llvm.ptr -> f64\n";
    std::string new_cum = fresh_ssa();
    out << "  " << new_cum << " = llvm.fadd " << nch_cum_var << ", " << ch_prob << " : f64\n";
    std::string hit = fresh_ssa();
    out << "  " << hit << " = llvm.fcmp \"olt\" " << roll << ", " << new_cum << " : f64\n";
    std::string not_yet = fresh_ssa();
    out << "  " << not_yet << " = llvm.fcmp \"oge\" " << roll << ", " << nch_cum_var << " : f64\n";
    std::string apply = fresh_ssa();
    out << "  " << apply << " = llvm.and " << hit << ", " << not_yet << " : i1\n";
    std::string lbl_apply = fresh_label("nch_apply");
    std::string lbl_skip = fresh_label("nch_skip");
    out << "  llvm.cond_br " << apply << ", ^" << lbl_apply << ", ^" << lbl_skip << "\n";
    out << "^" << lbl_apply << ":\n";
    // Pauli-frame XOR: tid0-only in coop (XOR is not idempotent). All threads
    // took the identical branch here (same RNG/roll), so guarding the mutation
    // to tid0 is safe; a barrier after makes the updated frame visible.
    {
        std::string xor_guard = cooperative_mode ? emit_tid0_guard_begin(out) : "";
        // Load and XOR x[0..1] into px, z[0..1] into pz
        for (int w = 0; w < (int)kPauliWords; ++w) {
            std::string x_off = emit_const_i64(out, w * 8);
            std::string x_ptr = fresh_ssa();
            out << "  " << x_ptr << " = llvm.getelementptr " << ch_base << "["
                << x_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
            std::string xval = fresh_ssa();
            out << "  " << xval << " = llvm.load " << x_ptr << " : !llvm.ptr -> i64\n";
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
        for (int w = 0; w < (int)kPauliWords; ++w) {
            std::string z_off = emit_const_i64(out, kPauliWords * 8 + w * 8);
            std::string z_ptr = fresh_ssa();
            out << "  " << z_ptr << " = llvm.getelementptr " << ch_base << "["
                << z_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
            std::string zval = fresh_ssa();
            out << "  " << zval << " = llvm.load " << z_ptr << " : !llvm.ptr -> i64\n";
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
        if (cooperative_mode) emit_tid0_guard_end_nobarrier(out, xor_guard);
    }
    out << "  llvm.br ^" << lbl_skip << "\n";
    out << "^" << lbl_skip << ":\n";
    // Continue channel loop
    std::string nch_next = fresh_ssa();
    out << "  " << nch_next << " = llvm.add " << nch_var << ", %c1_i64 : i64\n";
    out << "  llvm.br ^" << nch_hdr << "(" << nch_next << ", " << new_cum << " : i64, f64)\n";
    out << "^" << nch_done << ":\n";
    // Advance nni and draw next — on ALL threads to keep RNG/nni in sync.
    std::string next_si = emit_const_i32(out, site_idx + 1);
    out << "  llvm.store " << next_si << ", %nni_ptr : i32, !llvm.ptr\n";
    emit_draw_next_noise(out, flat);
    out << "  llvm.br ^" << lbl_skip_noise << "\n";
    out << "^" << lbl_skip_noise << ":\n";
    // Close the whole-site tid0 guard (no barrier here; the uniform barrier
    // below publishes tid0's frame writes to all threads).
    if (cooperative_mode) emit_tid0_guard_end_nobarrier(out, noise_t0);
    // Barrier so the tid0 frame XOR is visible to all threads before the next op.
    if (cooperative_mode) emit_barrier(out);
    break;
}
case Opcode::OP_NOISE_BLOCK: {
    // Redundant-full across coop threads (keep RNG/nni synced); only the
    // Pauli-frame XOR is tid0-guarded (see OP_NOISE rationale).
    // Process noise sites in [ins.a, ins.a + ins.b) with scheduling.
    //
    // IR-SIZE CRITICAL: emit a RUNTIME loop over the site index instead of
    // unrolling one full site block per site. Every site block is identical
    // except the constant `si`, and all site metadata is already loaded from
    // %noise_sites_ptr[si*16] at runtime — so we only need `si` to be a loop
    // variable. Unrolling produced ~100k lines / 32 MB IR for circuit_d5
    // (~1200 sites/block), which OOM/crashed llc. All 256 coop threads run this
    // loop in lockstep (per-thread private nni + RNG), so it stays uniform.
    uint32_t start = ins.a;
    uint32_t end_val = start + ins.b;
    uint32_t bound = std::min(end_val, (uint32_t)flat.noise_sites.size());
    out << "  // noise_block [" << start << ", " << end_val << ") (runtime loop, bound=" << bound << ")\n";
    if (bound > start) {
        // COOP: run the WHOLE site loop on tid0 only (matches gold
        // hip_sampler.hip:1698). All coop scalar RNG draws are tid0-only, so
        // tid0's per-thread nni + RNG stream stays aligned with SVM; the other
        // 255 threads did identical redundant work before — pure waste. The
        // trailing barrier (outside this guard) publishes tid0's frame writes.
        std::string nbl_t0 = cooperative_mode ? emit_tid0_guard_begin(out) : "";
        std::string nb_start_c = emit_const_i64(out, start);
        std::string nb_bound_c = emit_const_i64(out, bound);
        std::string nbl_hdr = fresh_label("nbl_hdr");
        std::string nbl_body = fresh_label("nbl_body");
        std::string nbl_exit = fresh_label("nbl_exit");
        std::string si64 = fresh_ssa();
        out << "  llvm.br ^" << nbl_hdr << "(" << nb_start_c << " : i64)\n";
        out << "^" << nbl_hdr << "(" << si64 << ": i64):\n";
        std::string nbl_cond = fresh_ssa();
        out << "  " << nbl_cond << " = llvm.icmp \"ult\" " << si64 << ", " << nb_bound_c << " : i64\n";
        out << "  llvm.cond_br " << nbl_cond << ", ^" << nbl_body << ", ^" << nbl_exit << "\n";
        out << "^" << nbl_body << ":\n";
        std::string si32 = fresh_ssa();
        out << "  " << si32 << " = llvm.trunc " << si64 << " : i64 to i32\n";
        std::string nni_cur = fresh_ssa();
        out << "  " << nni_cur << " = llvm.load %nni_ptr : !llvm.ptr -> i32\n";
        std::string nni_match = fresh_ssa();
        out << "  " << nni_match << " = llvm.icmp \"eq\" " << nni_cur << ", " << si32 << " : i32\n";
        std::string lbl_do = fresh_label("nb_do");
        std::string lbl_end = fresh_label("nb_end");
        out << "  llvm.cond_br " << nni_match << ", ^" << lbl_do << ", ^" << lbl_end << "\n";
        out << "^" << lbl_do << ":\n";
        // Load site metadata at %noise_sites_ptr[si*16] (runtime si)
        std::string sb_off = fresh_ssa();
        { std::string c16 = emit_const_i64(out, 16);
          out << "  " << sb_off << " = llvm.mul " << si64 << ", " << c16 << " : i64\n"; }
        std::string sb_base = fresh_ssa();
        out << "  " << sb_base << " = llvm.getelementptr %noise_sites_ptr["
            << sb_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        std::string nb_ch_offset = fresh_ssa();
        out << "  " << nb_ch_offset << " = llvm.load " << sb_base << " : !llvm.ptr -> i32\n";
        std::string nb_count_off = emit_const_i64(out, 4);
        std::string nb_count_ptr = fresh_ssa();
        out << "  " << nb_count_ptr << " = llvm.getelementptr " << sb_base << "["
            << nb_count_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        std::string nb_ch_count = fresh_ssa();
        out << "  " << nb_ch_count << " = llvm.load " << nb_count_ptr << " : !llvm.ptr -> i32\n";
        std::string nb_ps_off = emit_const_i64(out, 8);
        std::string nb_ps_ptr = fresh_ssa();
        out << "  " << nb_ps_ptr << " = llvm.getelementptr " << sb_base << "["
            << nb_ps_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        std::string nb_prob_sum = fresh_ssa();
        out << "  " << nb_prob_sum << " = llvm.load " << nb_ps_ptr << " : !llvm.ptr -> f64\n";
        std::string nb_u = emit_rng_uniform(out);
        std::string nb_roll = fresh_ssa();
        out << "  " << nb_roll << " = llvm.fmul " << nb_u << ", " << nb_prob_sum << " : f64\n";
        // Channel loop (same as OP_NOISE)
        std::string nb_ch_count_64 = fresh_ssa();
        out << "  " << nb_ch_count_64 << " = llvm.zext " << nb_ch_count << " : i32 to i64\n";
        std::string nb_ch_off_64 = fresh_ssa();
        out << "  " << nb_ch_off_64 << " = llvm.zext " << nb_ch_offset << " : i32 to i64\n";
        std::string nb_fzero = fresh_ssa();
        out << "  " << nb_fzero << " = llvm.mlir.constant(0.0 : f64) : f64\n";
        std::string nbch_var = fresh_ssa(), nbch_cum_var = fresh_ssa();
        std::string nbch_hdr = fresh_label("nbch_hdr"), nbch_body = fresh_label("nbch_body"), nbch_done = fresh_label("nbch_done");
        out << "  llvm.br ^" << nbch_hdr << "(%c0_i64, " << nb_fzero << " : i64, f64)\n";
        out << "^" << nbch_hdr << "(" << nbch_var << ": i64, " << nbch_cum_var << ": f64):\n";
        std::string nbch_cond = fresh_ssa();
        out << "  " << nbch_cond << " = llvm.icmp \"ult\" " << nbch_var << ", " << nb_ch_count_64 << " : i64\n";
        out << "  llvm.cond_br " << nbch_cond << ", ^" << nbch_body << ", ^" << nbch_done << "\n";
        out << "^" << nbch_body << ":\n";
        std::string nb_gci = fresh_ssa();
        out << "  " << nb_gci << " = llvm.add " << nb_ch_off_64 << ", " << nbch_var << " : i64\n";
        std::string nb_ch_stride = emit_const_i64(out, sizeof(GpuChannel));
        std::string nb_ch_boff = fresh_ssa();
        out << "  " << nb_ch_boff << " = llvm.mul " << nb_gci << ", " << nb_ch_stride << " : i64\n";
        std::string nb_ch_base = fresh_ssa();
        out << "  " << nb_ch_base << " = llvm.getelementptr %noise_channels_ptr["
            << nb_ch_boff << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        std::string nb_prob_off = emit_const_i64(out, offsetof(GpuChannel, prob));
        std::string nb_prob_ptr = fresh_ssa();
        out << "  " << nb_prob_ptr << " = llvm.getelementptr " << nb_ch_base << "["
            << nb_prob_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        std::string nb_ch_prob = fresh_ssa();
        out << "  " << nb_ch_prob << " = llvm.load " << nb_prob_ptr << " : !llvm.ptr -> f64\n";
        std::string nb_new_cum = fresh_ssa();
        out << "  " << nb_new_cum << " = llvm.fadd " << nbch_cum_var << ", " << nb_ch_prob << " : f64\n";
        std::string nb_hit = fresh_ssa();
        out << "  " << nb_hit << " = llvm.fcmp \"olt\" " << nb_roll << ", " << nb_new_cum << " : f64\n";
        std::string nb_not_yet = fresh_ssa();
        out << "  " << nb_not_yet << " = llvm.fcmp \"oge\" " << nb_roll << ", " << nbch_cum_var << " : f64\n";
        std::string nb_apply = fresh_ssa();
        out << "  " << nb_apply << " = llvm.and " << nb_hit << ", " << nb_not_yet << " : i1\n";
        std::string nb_lbl_apply = fresh_label("nbch_apply");
        std::string nb_lbl_skip = fresh_label("nbch_skip");
        out << "  llvm.cond_br " << nb_apply << ", ^" << nb_lbl_apply << ", ^" << nb_lbl_skip << "\n";
        out << "^" << nb_lbl_apply << ":\n";
        {
            std::string nb_xor_guard = cooperative_mode ? emit_tid0_guard_begin(out) : "";
            for (int w = 0; w < (int)kPauliWords; ++w) {
                std::string x_off = emit_const_i64(out, w * 8);
                std::string x_ptr = fresh_ssa();
                out << "  " << x_ptr << " = llvm.getelementptr " << nb_ch_base << "["
                    << x_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
                std::string xval = fresh_ssa();
                out << "  " << xval << " = llvm.load " << x_ptr << " : !llvm.ptr -> i64\n";
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
            for (int w = 0; w < (int)kPauliWords; ++w) {
                std::string z_off = emit_const_i64(out, kPauliWords * 8 + w * 8);
                std::string z_ptr = fresh_ssa();
                out << "  " << z_ptr << " = llvm.getelementptr " << nb_ch_base << "["
                    << z_off << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
                std::string zval = fresh_ssa();
                out << "  " << zval << " = llvm.load " << z_ptr << " : !llvm.ptr -> i64\n";
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
            if (cooperative_mode) emit_tid0_guard_end_nobarrier(out, nb_xor_guard);
        }
        out << "  llvm.br ^" << nb_lbl_skip << "\n";
        out << "^" << nb_lbl_skip << ":\n";
        std::string nbch_next = fresh_ssa();
        out << "  " << nbch_next << " = llvm.add " << nbch_var << ", %c1_i64 : i64\n";
        out << "  llvm.br ^" << nbch_hdr << "(" << nbch_next << ", " << nb_new_cum << " : i64, f64)\n";
        out << "^" << nbch_done << ":\n";
        std::string nb_next_si = fresh_ssa();
        out << "  " << nb_next_si << " = llvm.add " << si32 << ", %c1_i32 : i32\n";
        out << "  llvm.store " << nb_next_si << ", %nni_ptr : i32, !llvm.ptr\n";
        emit_draw_next_noise(out, flat);
        out << "  llvm.br ^" << lbl_end << "\n";
        out << "^" << lbl_end << ":\n";
        // loop increment + back-edge
        std::string si_next = fresh_ssa();
        out << "  " << si_next << " = llvm.add " << si64 << ", %c1_i64 : i64\n";
        out << "  llvm.br ^" << nbl_hdr << "(" << si_next << " : i64)\n";
        out << "^" << nbl_exit << ":\n";
        // Close the whole-block tid0 guard (no barrier here; the uniform
        // barrier below publishes tid0's frame writes to all threads).
        if (cooperative_mode) emit_tid0_guard_end_nobarrier(out, nbl_t0);
    }
    // Barrier so tid0 frame XORs are visible to all threads before the next op.
    if (cooperative_mode) emit_barrier(out);
    break;
}
case Opcode::OP_READOUT_NOISE: {
    // Readout noise: flip measurement bit with probability.
    // COOP: the scalar RNG draw + meas flip MUST be tid0-only (matches gold
    // hip_sampler.hip:1722). All coop scalar RNG draws are tid0-only so tid0's
    // stream stays byte-aligned with the SVM single-thread stream; a
    // redundant-full draw here desyncs tid0 vs the other 255 threads (tid0 is
    // already ahead from tid0-only sample_branch draws) and corrupts shots >=1.
    if (ins.a < flat.readout_noise.size()) {
        const auto& rn = flat.readout_noise[ins.a];
        std::string rn_guard = cooperative_mode ? emit_tid0_guard_begin(out) : "";
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
        if (cooperative_mode) emit_tid0_guard_end(out, rn_guard);
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
    // Classically-controlled Pauli (feedforward correction): apply the mask to
    // the Pauli frame ONLY when the controlling measurement meas[ins.b] != 0.
    // GPU-SVM: apply_pauli() early-returns if st.meas[condition_idx]==0
    // (hip_sampler.hip). Applying unconditionally corrupts the frame
    // deterministically -> wrong stabilizer measurements -> wrong observable.
    if (ins.a < flat.pauli_masks.size()) {
        const auto& mask = flat.pauli_masks[ins.a];
        // Load controlling measurement bit meas[ins.b]
        std::string cidx = emit_const_i64(out, ins.b);
        std::string cptr = fresh_ssa();
        out << "  " << cptr << " = llvm.getelementptr inbounds %meas_ptr["
            << cidx << "] : (!llvm.ptr, i64) -> !llvm.ptr, i8\n";
        std::string cval = fresh_ssa();
        out << "  " << cval << " = llvm.load " << cptr << " : !llvm.ptr -> i8\n";
        std::string cnz = fresh_ssa();
        out << "  " << cnz << " = llvm.icmp \"ne\" " << cval << ", %c0_i8 : i8\n";
        std::string lbl_apply = fresh_label("ap_apply");
        std::string lbl_done = fresh_label("ap_done");
        out << "  llvm.cond_br " << cnz << ", ^" << lbl_apply << ", ^" << lbl_done << "\n";
        out << "^" << lbl_apply << ":\n";
        // Frame mutation: tid0-only in coop (shared frame); barrier after.
        std::string ap_guard = cooperative_mode ? emit_tid0_guard_begin(out) : "";
        for (int w = 0; w < (int)kPauliWords; ++w) {
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
        if (cooperative_mode) emit_tid0_guard_end_nobarrier(out, ap_guard);
        out << "  llvm.br ^" << lbl_done << "\n";
        out << "^" << lbl_done << ":\n";
        if (cooperative_mode) emit_barrier(out);
    }
    break;
}
