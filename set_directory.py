import os
#
# result_path = "./results"
def set_directory(result_path = "./results"):
    # results
    if not os.path.isdir(result_path): os.mkdir(result_path)
    # if not os.path.isdir(os.path.join(result_path, "trends")): os.mkdir(os.path.join(result_path, "trends"))
    # if not os.path.isdir(os.path.join(result_path, "outer_loop")): os.mkdir(os.path.join(result_path, "outer_loop"))
    #
    # # CV folders
    # if not os.path.isdir(os.path.join(result_path, "outer_loop", "CV1")): os.mkdir(os.path.join(result_path, "outer_loop", "CV1"))
    # if not os.path.isdir(os.path.join(result_path, "outer_loop", "CV2")): os.mkdir(os.path.join(result_path, "outer_loop", "CV2"))
    # if not os.path.isdir(os.path.join(result_path, "outer_loop", "CV3")): os.mkdir(os.path.join(result_path, "outer_loop", "CV3"))
    # if not os.path.isdir(os.path.join(result_path, "outer_loop", "CV4")): os.mkdir(os.path.join(result_path, "outer_loop", "CV4"))
    #
    # if not os.path.isdir(os.path.join(result_path, "outer_loop", "CV1", "graph")): os.mkdir(os.path.join(result_path, "outer_loop", "CV1", "graph"))
    # if not os.path.isdir(os.path.join(result_path, "outer_loop", "CV2", "graph")): os.mkdir(os.path.join(result_path, "outer_loop", "CV2", "graph"))
    # if not os.path.isdir(os.path.join(result_path, "outer_loop", "CV3", "graph")): os.mkdir(os.path.join(result_path, "outer_loop", "CV3", "graph"))
    # if not os.path.isdir(os.path.join(result_path, "outer_loop", "CV4", "graph")): os.mkdir(os.path.join(result_path, "outer_loop", "CV4", "graph"))
    #
    # if not os.path.isdir(os.path.join(result_path, "outer_loop", "CV1", "inner_loop")): os.mkdir(os.path.join(result_path, "outer_loop", "CV1", "inner_loop"))
    # if not os.path.isdir(os.path.join(result_path, "outer_loop", "CV2", "inner_loop")): os.mkdir(os.path.join(result_path, "outer_loop", "CV2", "inner_loop"))
    # if not os.path.isdir(os.path.join(result_path, "outer_loop", "CV3", "inner_loop")): os.mkdir(os.path.join(result_path, "outer_loop", "CV3", "inner_loop"))
    # if not os.path.isdir(os.path.join(result_path, "outer_loop", "CV4", "inner_loop")): os.mkdir(os.path.join(result_path, "outer_loop", "CV4", "inner_loop"))
    #
    # for ind_outer in range(1,4+1):
    #     for ind_inner in range(1, 4+1):
    #         if not os.path.isdir(os.path.join(result_path, "outer_loop", "CV"+str(ind_outer), "inner_loop", "CV"+str(ind_inner))):\
    #             os.mkdir(os.path.join(result_path, "outer_loop", "CV"+str(ind_outer), "inner_loop", "CV"+str(ind_inner)))
