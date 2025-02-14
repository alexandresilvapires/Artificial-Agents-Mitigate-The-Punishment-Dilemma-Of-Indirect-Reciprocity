include("./utils.jl")
include("./reputation.jl")
include("./strategy.jl")
include("./plotter.jl")

# Space to run statistics. 
# Define all the parameters below and in the end of the file choose the desired function to run

# Errors
const execError::Float64 = 0.01
const assessError::Float64 = 0.01

# Common social norms
const snIS::Vector{Float64} = add_errors_sn(utils.snIS_noerrors, execError, assessError)
const snSJ::Vector{Float64} = add_errors_sn(utils.snSJ_noerrors, execError, assessError)
const snSH::Vector{Float64} = add_errors_sn(utils.snSH_noerrors, execError, assessError)
const snSS::Vector{Float64} = add_errors_sn(utils.snSS_noerrors, execError, assessError)
const snAG::Vector{Float64} = [1.0, 1.0, 1.0, 1.0]
const snAB::Vector{Float64} = [0.0, 0.0, 0.0, 0.0]

# Each vector is a possible setup of SNs to be used in the population. Some functions override this depending on what they study
const snsIS::Vector{Vector{Float64}} = [snIS,snIS,snIS,snIS,snIS] #snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA
const snsSJ::Vector{Vector{Float64}} = [snSJ,snSJ,snSJ,snSJ,snSJ] #snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA
const snsSH::Vector{Vector{Float64}} = [snSH,snSH,snSH,snSH,snSH] #snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA
const snsSS::Vector{Vector{Float64}} = [snSS,snSS,snSS,snSS,snSS] #snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA
const snsAG::Vector{Vector{Float64}} = [snAG,snAG,snAG,snAG,snAG] #snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA
const snsAB::Vector{Vector{Float64}} = [snAB,snAB,snAB,snAB,snAB] #snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA

# Norm studies for AAs
const norms = [snsIS, snsSJ, snsSH, snsSS, snsAG,snsAB]
const norm_names = ["Image Score","Stern-Judging","Shunning","Simple Standing","All Good","All Bad"]

# Strategy study for AAs
sn_all_same::Vector{Vector{Float64}} = [snSH, snSH, snSH, snSH, snSH]
const stratsAA = [utils.STRAT_ALLC, utils.STRAT_ALLD, utils.STRAT_DISC]
const strat_names = ["ALLC","ALLD","DISC"]

const valsInteractionAA::Vector{Float64} = collect(0.0:0.02:1.0)

# AA Strategy
const stratAA::Int = utils.STRAT_DISC                                   # The strategy used by the AA. utils.STRAT_DISC ; utils.STRAT_ALLC ; utils.STRAT_ALLD
const AA_POP = (stratAA == utils.STRAT_ALLC ? "AllC" : (stratAA == utils.STRAT_ALLD ? "AllD" : "Disc"))

const imitAAs::Bool = false                                             # If true, humans can imitate the AA's strategy

const popsize::Int = 100
const b::Float64 = 3.0                                                  # c = 1 always. Some functions override this, such as b_study

const mutChance::Float64 = 1.0 / popsize
const gossipRounds::Int = 0                                             # Frequency of gossip based on A mechanistic model of gossip, reputations, and cooperation. All experiments are ran with the value of 0
const strenghtOfSelection::Float64 = 1.0

const justGenPlots::Bool = false                                        # if true, instead of generating data and plotting it, it just accesses the folder to generate the plots

foldername = "test"                                                     # change folder name for each experiment
plotpath = "Plots/"*foldername

# Premade functions for studies below

# Study behaviour with varying levels of AAs and a fixed b/c
function AA_study(filenames::Vector{String})
    for norm_i in eachindex(norms)
        !justGenPlots && vary_variable([norms[norm_i], 0.0, stratAA, popsize, execError, assessError, b, mutChance, gossipRounds, strenghtOfSelection,imitAAs],"interactionsAA", 2, valsInteractionAA, 
            filenames[norm_i], foldername)
    end

    plot_grad(stratAA, norms, [0,0.05,0.1,0.15,0.2], popsize, execError, b, gossipRounds, mutChance, imitAAs, plotpath, filenames)

    run_all_plots(plotpath, filenames, filenames, "Social Norm")
end

# Study how b/c changes cooperation for a fixed level of AAs
function b_study(interactAA::Float64=0.0, filenames::Vector{String}=norm_names)
    bs::Vector{Float64} = [1+i for i in 0:0.2:7]
    for norm_i in eachindex(norms)
        !justGenPlots && vary_variable([norms[norm_i], interactAA, stratAA, popsize, execError, assessError, b, mutChance, gossipRounds, strenghtOfSelection,imitAAs],"b", 7, bs, 
            filenames[norm_i], foldername)
    end
    
    b_study_plot(plotpath, filenames, "Social Norm")
end

# Study Populations containing only a fixed social norm and varying the strategy of the AA
function AA_strat_study(filenames::Vector{String})
    for strat_i in eachindex(stratsAA)
        !justGenPlots && vary_variable([sn_all_same, 0.0, stratsAA[strat_i], popsize, execError, assessError, b, mutChance, gossipRounds, strenghtOfSelection,imitAAs],"interactionsAA", 2, valsInteractionAA, 
            filenames[strat_i], foldername)
    end
    run_all_plots(plotpath, filenames, filenames, "AA Strategy")
end

# Study Populations containing only a fixed social norm and vary b/c and the amount of AAs
function b_AA_study(simplifiedAAjudgement::Bool=false)
    norms_to_try = [snIS, snSJ, snSH, snSS]
    norm_tags = ["IS","SJ","SH","SS"]

    bs::Vector{Float64} = [1+i for i in 0:0.2:7]
    AAs::Vector{Float64} = [0+i for i in 0:0.1:0.5]

    filenames::Vector{String} = [string(i) for i in AAs]

    totalfilenames = []
    
    for i in eachindex(norms_to_try)
        sn_all_equal = [norms_to_try[i] for k in 1:5]

        if simplifiedAAjudgement
            sn_all_equal = [norms_to_try[i], snIS, norms_to_try[i], norms_to_try[i], norms_to_try[i]]
        end

        f_names = []
        for AAs_i in eachindex(AAs)
            name = norm_tags[i]*"_"*filenames[AAs_i]
            !justGenPlots && vary_variable([sn_all_equal, AAs[AAs_i], stratAA, popsize, execError, assessError, b, mutChance, gossipRounds, strenghtOfSelection,imitAAs],"b", 7, bs, name, foldername)
            push!(f_names, name)
        end
        push!(totalfilenames, f_names)
    end
    b_study_multiple_plot(plotpath, totalfilenames, "τ", norm_tags, AAs)
end

# Study behaviour with varying levels of AAs for all combinations of second order norms in the paper
# Where nH_H_H = snH_H_AA, snAA_H_H=snAA_H_AA
function AA_study_all_combinations(filenames::Vector{String}, only_equal_norms::Bool=false)
    all_norms = [snIS, snSJ, snSH, snSS, snAG, snAB]
    norm_tags = ["IS","SJ","SH","SS", "AG", "AB"]

    for AAstratToUse in eachindex(stratsAA)
        # for each strategy for the AA
        # we test all equal norms, plus all equal norms except H-A being IS
        # plus all social norms except A-H being all other norms
        # and all social norms except A-H being all other and H-A being IS
        # Naming convention is _AASTRAT_H-A=NORM_A-H=NORM
        if (!only_equal_norms)
            iteration_counter = 0
            total_iterations = length(norms) * 2
            for norm_a_h in eachindex(norms)
                for norm_h_a in [1,2]   # where 1 = use whatever norm_h_h is, and 2 = use IS
                    text_norm_h_a = norm_h_a == 1 ? "H-H" : "IS"
                    # folders always show the 6 human norms changing
                    
                    # change folder name for each experiment
                    foldername = "_"*strat_names[AAstratToUse]*"_"*"H-A="*text_norm_h_a*"_A-H="*norm_tags[norm_a_h]
                    plotpath = "Plots/"*foldername
                    
                    # Counter for info
                    iteration_counter += 1
                    global_index = iteration_counter
                    progress_percentage = (global_index / total_iterations) * 100
                    println("Currently on: "*foldername*" Progress: $progress_percentage%")
                    
                    norms_to_use = []
                    # snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA
                    for norm_h_h in eachindex(norms)
                        snH_H = all_norms[norm_h_h]
                        snH_AA = norm_h_a == 1 ? snH_H : snIS
                        snAA_H = all_norms[norm_a_h]
                        push!(norms_to_use,[snH_H, snH_AA, snH_H, snAA_H, snAA_H])
                    end
                    for norm_i in eachindex(norms_to_use)
                        !justGenPlots && vary_variable([norms_to_use[norm_i], 0.0, stratsAA[AAstratToUse], popsize, execError, assessError, b, mutChance, gossipRounds, strenghtOfSelection,imitAAs],"interactionsAA", 2, valsInteractionAA, filenames[norm_i], foldername)
                    end
                    plot_grad(stratsAA[AAstratToUse], norms_to_use, [0,0.05,0.1,0.15,0.2], popsize, execError, b, gossipRounds, mutChance, imitAAs, plotpath, filenames)
                    run_all_plots(plotpath, filenames, filenames, "Social Norm")
                end
            end
        end
        # combine the all equal norms in the same place
        # change folder name for each experiment
        for useHA_IS in [false,true]    # run all equal twice, once with sn_H_A = IS
            if (useHA_IS && only_equal_norms) continue end
            add_on = ""
            if (useHA_IS) add_on = "_H-A=IS" end
            foldername = "_"*strat_names[AAstratToUse]*"_allequal"*add_on
            plotpath = "Plots/"*foldername
            norms_to_use = []
            # snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AAs
            for n in all_norms
                ha_norm = n
                if (useHA_IS) ha_norm = snIS end
                push!(norms_to_use, [n, ha_norm, n, n, n])
            end
            
            for norm_i in eachindex(norms_to_use)
                !justGenPlots && vary_variable([norms_to_use[norm_i], 0.0, stratsAA[AAstratToUse], popsize, execError, assessError, b, mutChance, gossipRounds, strenghtOfSelection,imitAAs],"interactionsAA", 2, valsInteractionAA, filenames[norm_i], foldername)
            end
            plot_grad(stratsAA[AAstratToUse], norms_to_use, [0,0.05,0.1,0.15,0.2], popsize, execError, b, gossipRounds, mutChance, imitAAs, plotpath, filenames)
            run_all_plots(plotpath, filenames, filenames, "Social Norm")
        end
    end
end

# Study how errors influence cooperation and disagreement
function error_study(filenames::Vector{String}=norm_names, HA_IS::Bool=false)
    
    # for each norm, we make two plots: one varying the exec and another the assessment error.
    # Each plot has a line for a varying level of tau
    
    error_values::Vector{Float64} = utils.generate_log_spaced_values(0.0, 0.5, 40)
    taus_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    allnorms = [utils.snIS_noerrors, utils.snSJ_noerrors, utils.snSH_noerrors, utils.snSS_noerrors]
    norm_tags = ["IS","SJ","SH","SS"]

    all_filenames = []  # keep all the final filenames in a structure like [[[IS_exec],[IS_assess]], [[.._exec],[.._assess]]]

    for norm_i in eachindex(allnorms)

        general_filenames = [filenames[norm_i] * "_" * string(tau) for tau in taus_to_test]

        ha_norm = (HA_IS ? utils.snIS_noerrors : allnorms[norm_i])
        totalnorm = [allnorms[norm_i], ha_norm, allnorms[norm_i], allnorms[norm_i], allnorms[norm_i]]  #snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA

        println("Current norm:"*filenames[norm_i])
        new_norm_filenames = [[],[]]
        new_norm_filenames[1] = [filename * "_exec" for filename in general_filenames]
        new_norm_filenames[2] = [filename * "_assess" for filename in general_filenames]
        
        for tau_i in eachindex(taus_to_test)
            # first test exec errors

            # We make various lines, with each tau
            !justGenPlots && vary_variable([totalnorm, taus_to_test[tau_i], stratAA, popsize, 0.0, 0.0, b, mutChance, gossipRounds, strenghtOfSelection,imitAAs],"execError", 5, error_values, new_norm_filenames[1][tau_i], foldername, true)

            # then test assessment errors
            # We make various lines, with each tau
            !justGenPlots && vary_variable([totalnorm, taus_to_test[tau_i], stratAA, popsize, 0.0, 0.0, b, mutChance, gossipRounds, strenghtOfSelection,imitAAs],"assessError", 6, error_values, new_norm_filenames[2][tau_i], foldername,true)
        end
        push!(all_filenames, new_norm_filenames)

    end
    # make a plot for each norm
    error_study_plot(plotpath, taus_to_test, error_values, all_filenames, norm_tags, "τ")
end

# Study cooperation and disagreement when interpolating between social norms social norms
function intermediate_norms_study()
    
    # make a matrix where we interpolate between (1, 0, a, b), a,b in [0, 1]
    taus_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    bcs_to_test = [3.0, 5.0]

    allnorms = [snIS, snSJ, snSH, snSS]
    allnorms_names = ["IS", "SJ", "SH","SS"]

    simplify_plot = false       # If true, simplifies the plot to remove numbers and expand norm names

    # Run once for all-equal, then run 4 times varying only H-A-H
    for tau in taus_to_test
        for bc in bcs_to_test
            println("----running tau="*string(tau)*"_bc="*string(bc))
            
            # Run when all norms are the same
            all_sets_of_norms = vec([[[1, 0, a, b], [1, 0, a, b], [1, 0, a, b], [1, 0, a, b], [1, 0, a, b]] for (a, b) in Iterators.product(0:0.1:1.0, 0:0.1:1.0)])

            fname = "intermediatenorms_tau="*string(tau)*"_bc="*string(bc)
            !justGenPlots && vary_variable([[snIS,snIS,snIS,snIS,snIS], tau, stratAA, popsize, execError, assessError, bc, mutChance, gossipRounds, strenghtOfSelection,imitAAs],"sns", 1, all_sets_of_norms, fname, foldername, true)
        
            intermediate_norms_plot(plotpath, fname, "H-H = A-H = H-A = (1, 0, α¹, α²)", fname,all_sets_of_norms, simplify_plot)
            println("allequal done")

            if (tau > 0.0)
                # 4 times using each norm and only varying H-A-H
                for n in eachindex(allnorms)
                    # snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA
                    all_sets_of_norms = vec([[allnorms[n], [1, 0, a, b], allnorms[n], allnorms[n], allnorms[n]] for (a, b) in Iterators.product(0:0.1:1.0, 0:0.1:1.0)])
                    
                    fname = "intermediatenorms_tau="*string(tau)*"_bc="*string(bc)*"_HAH=IS_OTHERS="*allnorms_names[n]
                    !justGenPlots && vary_variable([[snIS,snIS,snIS,snIS,snIS], tau, stratAA, popsize, execError, assessError, bc, mutChance, gossipRounds, strenghtOfSelection,imitAAs],"sns", 1, all_sets_of_norms, fname, foldername,true)
                    
                    intermediate_norms_plot(plotpath, fname, "H-H = A-H = "*allnorms_names[n]*", H-A = (1, 0, α¹, α²)",fname,all_sets_of_norms, simplify_plot)
                    println(allnorms_names[n]*" done")
                end
            end
        end
    end

    # make a plot for each norm
end

# Leave the desired function uncommented

#b_study(0.3)
AA_study(norm_names)
#AA_strat_study(strat_names)
#b_AA_study(true)
#AA_study_all_combinations(norm_names)
#error_study(norm_names, false)
#intermediate_norms_study()