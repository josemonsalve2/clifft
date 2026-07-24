; ModuleID = 'v2coop2.c'
source_filename = "v2coop2.c"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@lds_v = hidden local_unnamed_addr addrspace(3) global [1024 x float] zeroinitializer, align 4

; Function Attrs: convergent nofree norecurse nounwind
define hidden amdgpu_kernel void @coop_probe(ptr noundef writeonly captures(none) %g, i32 noundef %n) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %umax = tail call i32 @llvm.umax.i32(i32 %0, i32 768)
  %1 = add nuw nsw i32 %umax, 255
  %2 = sub nsw i32 %1, %0
  %3 = lshr i32 %2, 8
  %n.rnd.up = add nuw nsw i32 %3, 2
  %n.vec = and i32 %n.rnd.up, 33554430
  %broadcast.splatinsert = insertelement <2 x i32> poison, i32 %3, i64 0
  %broadcast.splat = shufflevector <2 x i32> %broadcast.splatinsert, <2 x i32> poison, <2 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %pred.store.continue33, %entry
  %index = phi i32 [ 0, %entry ], [ %index.next, %pred.store.continue33 ]
  %4 = shl i32 %index, 8
  %offset.idx = add i32 %0, %4
  %broadcast.splatinsert30 = insertelement <2 x i32> poison, i32 %index, i64 0
  %broadcast.splat31 = shufflevector <2 x i32> %broadcast.splatinsert30, <2 x i32> poison, <2 x i32> zeroinitializer
  %vec.iv = or disjoint <2 x i32> %broadcast.splat31, <i32 0, i32 1>
  %5 = icmp ule <2 x i32> %vec.iv, %broadcast.splat
  %6 = extractelement <2 x i1> %5, i64 0
  br i1 %6, label %pred.store.if, label %pred.store.continue

pred.store.if:                                    ; preds = %vector.body
  %7 = getelementptr inbounds nuw [1024 x float], ptr addrspace(3) @lds_v, i32 0, i32 %offset.idx
  %8 = uitofp nneg i32 %offset.idx to float
  store float %8, ptr addrspace(3) %7, align 4, !tbaa !4
  br label %pred.store.continue

pred.store.continue:                              ; preds = %pred.store.if, %vector.body
  %9 = extractelement <2 x i1> %5, i64 1
  br i1 %9, label %pred.store.if32, label %pred.store.continue33

pred.store.if32:                                  ; preds = %pred.store.continue
  %10 = add i32 %offset.idx, 256
  %11 = getelementptr inbounds nuw [1024 x float], ptr addrspace(3) @lds_v, i32 0, i32 %10
  %12 = uitofp nneg i32 %10 to float
  store float %12, ptr addrspace(3) %11, align 4, !tbaa !4
  br label %pred.store.continue33

pred.store.continue33:                            ; preds = %pred.store.if32, %pred.store.continue
  %index.next = add nuw i32 %index, 2
  %13 = icmp eq i32 %index.next, %n.vec
  br i1 %13, label %for.cond.cleanup, label %vector.body, !llvm.loop !8

for.cond.cleanup:                                 ; preds = %pred.store.continue33
  tail call void @llvm.amdgcn.s.barrier()
  %and = and i32 %0, 63
  %arrayidx3 = getelementptr inbounds nuw [1024 x float], ptr addrspace(3) @lds_v, i32 0, i32 %0
  %14 = load float, ptr addrspace(3) %arrayidx3, align 4, !tbaa !4
  %15 = bitcast float %14 to i32
  %xor = shl nuw nsw i32 %and, 2
  %shl = xor i32 %xor, 128
  %16 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %shl, i32 %15)
  %17 = bitcast i32 %16 to float
  %add10 = fadd float %14, %17
  %18 = bitcast float %add10 to i32
  %xor.1 = shl nuw nsw i32 %and, 2
  %shl.1 = xor i32 %xor.1, 64
  %19 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %shl.1, i32 %18)
  %20 = bitcast i32 %19 to float
  %add10.1 = fadd float %add10, %20
  %21 = bitcast float %add10.1 to i32
  %xor.2 = shl nuw nsw i32 %and, 2
  %shl.2 = xor i32 %xor.2, 32
  %22 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %shl.2, i32 %21)
  %23 = bitcast i32 %22 to float
  %add10.2 = fadd float %add10.1, %23
  %24 = bitcast float %add10.2 to i32
  %xor.3 = shl nuw nsw i32 %and, 2
  %shl.3 = xor i32 %xor.3, 16
  %25 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %shl.3, i32 %24)
  %26 = bitcast i32 %25 to float
  %add10.3 = fadd float %add10.2, %26
  %27 = bitcast float %add10.3 to i32
  %xor.4 = shl nuw nsw i32 %and, 2
  %shl.4 = xor i32 %xor.4, 8
  %28 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %shl.4, i32 %27)
  %29 = bitcast i32 %28 to float
  %add10.4 = fadd float %add10.3, %29
  %30 = bitcast float %add10.4 to i32
  %xor.5 = shl nuw nsw i32 %and, 2
  %shl.5 = xor i32 %xor.5, 4
  %31 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %shl.5, i32 %30)
  tail call void @llvm.amdgcn.s.barrier()
  %cmp13 = icmp eq i32 %0, 0
  %cmp15 = icmp ne i32 %n, 0
  %or.cond = and i1 %cmp13, %cmp15
  br i1 %or.cond, label %if.then, label %if.end

if.then:                                          ; preds = %for.cond.cleanup
  %g.global = addrspacecast ptr %g to ptr addrspace(1)
  %32 = bitcast i32 %31 to float
  %add10.5 = fadd float %add10.4, %32
  store float %add10.5, ptr addrspace(1) %g.global, align 4, !tbaa !4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.cond.cleanup
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #2

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umax.i32(i32, i32) #4

attributes #0 = { convergent nofree norecurse nounwind "amdgpu-agpr-alloc"="0" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx950" "target-features"="+16-bit-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+bf8-cvt-scale-insts,+bitop3-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot12-insts,+dot13-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+f16bf16-to-fp6bf6-cvt-scale-insts,+f32-to-f16bf16-cvt-sr-insts,+fp4-cvt-scale-insts,+fp6bf6-cvt-scale-insts,+fp8-conversion-insts,+fp8-cvt-scale-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+gfx950-insts,+mai-insts,+permlane16-swap,+permlane32-swap,+prng-inst,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="false" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #3 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{!"clang version 21.0.0git (https://github.com/llvm/llvm-project.git 20d6375796073f6a0f0ea6abe05ce454a3d617ff)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = distinct !{!8, !9, !10}
!9 = !{!"llvm.loop.mustprogress"}
!10 = !{!"llvm.loop.isvectorized", i32 1}
