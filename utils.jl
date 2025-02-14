module utils

using Memoization
using SparseArrays
using ArnoldiMethod
using Dates
using Serialization
using LinearAlgebra

# --------------------------
# Strat labels
# --------------------------

const STRAT_ALLC::Int = 0
const STRAT_ALLD::Int = 1
const STRAT_DISC::Int = 2

# --------------------------
# Default norms
# --------------------------

# Common social norms
const snIS_noerrors::Vector{Float64} = [1.0, 0.0, 1.0, 0.0]
const snSJ_noerrors::Vector{Float64} = [1.0, 0.0, 0.0, 1.0]
const snSH_noerrors::Vector{Float64} = [1.0, 0.0, 0.0, 0.0]
const snSS_noerrors::Vector{Float64} = [1.0, 0.0, 1.0, 1.0]
const snAG::Vector{Float64} = [1.0, 1.0, 1.0, 1.0]
const snAB::Vector{Float64} = [0.0, 0.0, 0.0, 0.0]

# --------------------------
# Memoization
# --------------------------

function clear_memoization_list(functions)
    for f in functions
        Memoization.empty_cache!(f)
    end
end

# --------------------------
# Plot folder making and parameter and results input
# --------------------------

function make_plot_folder(foldername::String = "")::String
    # Get current date and hour
    current_date_hour = Dates.now()

    # Create a formatted string for the folder name
    isdir("Plots") || mkdir("Plots")
    folder_name = (foldername == "" ? Dates.format(current_date_hour, "#Y#m#d_#H#M") : foldername)
    
    # Create the path for the folder
    folder_path = joinpath("Plots", folder_name)
    isdir(folder_path) || mkdir(folder_path)
    
    # Create the folder if it doesn't exist
    isdir(folder_path) || mkdir(folder_path)

    return folder_path
end

function write_parameters(folder_path::String, sns::Vector, interactionsAA::Float64, stratAA::Int, popsize::Int, 
            execError::Float64, assessError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int, strengthOfSelection::Float64, parameterVaried::String, rangeOfValues::Vector, imitAAs::Bool)
    parameters = (
        "Population Size" => popsize,
        "Frac. Interaction AA" => interactionsAA,
        "Strat AA" => stratAA,
        "Imit AAs" => imitAAs,
        "snH_H_H" => sns[1],
        "snH_AA_H" => sns[2],
        "snH_H_AA" => sns[3],
        "snAA_H_H" => sns[4],
        "snAA_H_AA" => sns[5],
        "errorExecut" => execError,
        "errorAssess" => assessError,
        "strengthOfSel" => strengthOfSelection,
        "mutationChance" => mutChance,
        "b/c" => b,
        "Gossip Rounds" => gossipRounds,
        "----------" => "",
        "Parameter Varied" => parameterVaried,
        "Range of Values" => rangeOfValues
    )
    write_parameters_txt(joinpath(folder_path, "parameters.txt"),parameters)
end

function write_parameters_txt(path::AbstractString, parameters)
    try
        open(path, "w") do file
            for (param, value) in parameters
                println(file, "$param = $value")
            end
        end
        println("Parameters written to $path\n")
    catch e
        println("Error writing parameters: $e\n")
    end
end

# Abstract method to write any result in the format A -> B in a file
function write_result_txt(path::String, tag::String, result)
    try
        open(path, "a") do file
            println(file, "$tag -> $result")
        end
    catch e
        println("Error writing data: $e\n")
    end
end

# Write all results straight from an array of strategy.get_all_data() to the respective files
# Since this is to be called for each time the results are calculated, it appends to the existing files
function write_all_results(path::String, tag::String, allResults::Vector, save_txt::Bool=false)

    # Create the path for the folder and create the folder if it doesn't exist
    resultsPath = joinpath(path, "Results")
    isdir(resultsPath) || mkdir(resultsPath)

    # Serialize results vector into a folder
    savedVarPath = joinpath(resultsPath, "ResultsBackup")
    isdir(savedVarPath) || mkdir(savedVarPath)
    fileBackup =  open(joinpath(savedVarPath, tag*"_results.jls"), "w")
    serialize(fileBackup, allResults)
    close(fileBackup)

        if (save_txt)
        # Go over each of the vars to make txts with all data
        valuesPath = joinpath(resultsPath, "CoopIndex")
        isdir(valuesPath) || mkdir(valuesPath)
        val = [data[1][1] for data in allResults]
        write_result_txt(joinpath(valuesPath, "CoopIndex_Avg.txt"), tag, val)
        val = [data[1][2] for data in allResults]
        write_result_txt(joinpath(valuesPath, "CoopIndex_All.txt"), tag, val)

        valuesPath = joinpath(resultsPath, "Reputation")
        isdir(valuesPath) || mkdir(valuesPath)
        val = [data[2][1] for data in allResults]
        write_result_txt(joinpath(valuesPath, "Reputation_Avg.txt"), tag, val)
        val = [data[2][2] for data in allResults]
        write_result_txt(joinpath(valuesPath, "Reputation_All.txt"), tag, val)

        valuesPath = joinpath(resultsPath, "StatDist")
        isdir(valuesPath) || mkdir(valuesPath)
        val = [data[3] for data in allResults]
        write_result_txt(joinpath(valuesPath, "StatDist.txt"), tag, val)
        val = [data[4] for data in allResults]
        write_result_txt(joinpath(valuesPath, "State_Avg.txt"), tag, val)

        valuesPath = joinpath(resultsPath, "Agreement")
        isdir(valuesPath) || mkdir(valuesPath)
        val = [data[5][1] for data in allResults]
        write_result_txt(joinpath(valuesPath, "Agreement_Avg.txt"), tag, val)
        val = [data[5][2] for data in allResults]
        write_result_txt(joinpath(valuesPath, "Agreement_All.txt"), tag, val)

        valuesPath = joinpath(resultsPath, "GradientOfSelection")
        isdir(valuesPath) || mkdir(valuesPath)
        val = [data[6] for data in allResults]
        write_result_txt(joinpath(valuesPath, "Gradients.txt"), tag, val)
    end
end

function read_parameters(file_path::AbstractString)
    params = Dict{AbstractString, Any}()

    open(file_path, "r") do file
        for line in eachline(file)
            parts = split(line, "=")
            key = strip(parts[1])
            value = try
                parse(Float64, strip(parts[2]))
            catch
                parse.(Float64, split(strip(parts[2]), ","))
            end
            params[key] = value
        end
    end

    return params
end

# --------------------------
# Result extraction from txt
# --------------------------

# Converts a coopIndex.txt to a vector of vectors with the cooperation index of each
function process_results(filepath::String, population::String)::Vector{Vector{Float64}}
    result_arrays = Vector{Vector{Float32}}()
    open(filepath) do file
        for line in eachline(file)
            if startswith(line, population)
                # Extract the array part from the line
                array_part = split(line, " -> ")[2]
                # Remove "Float32[" and "]" from the array part
                array_string = replace(array_part, r"Float64\[|\]" => "")
                # Convert the comma-separated string into an array of Float64
                array = parse.(Float32, split(array_string, ", "))
                push!(result_arrays, array)
            end
        end
    end
    return result_arrays
end

function deserialize_file(file_path::AbstractString)
    # Open the file for reading
    file = open(file_path, "r")
    
    # Deserialize the content of the file
    data = deserialize(file)
    
    # Close the file
    close(file)
    
    return data
end

# --------------------------
# Processing results
# --------------------------

function get_average_rep_states(results, popsize)
    # Receives the result straight from get_all_data
    # Returns the average human reputation for each of the states given the proportion of each strategy
    # returns [avgRepState1, avgRepState2, ...]

    function get_states_strat(popsize::Int)::Vector{Tuple{Int, Int, Int}}
        return [(nAllC, nAllD, popsize - nAllC - nAllD) for nAllC in 0:popsize for nAllD in 0:(popsize - nAllC) if nAllC + nAllD <= popsize]
    end

    states = get_states_strat(popsize)
    avg_reps = []

    for i in eachindex(states)
        rep_state = results[2][2][i]
        arep = (rep_state[1]*states[i][1] + rep_state[2]*states[i][2] + rep_state[3]*states[i][3]) / popsize
        push!(avg_reps, arep)
    end

    return avg_reps
end

# --------------------------
# Parameter extraction from txt
# --------------------------

# Get parameter value from parameter.txt
function get_parameter_value(filepath::String, parameter::String)
    value = ""
    open(joinpath(filepath, "parameters.txt")) do file
        for line in eachline(file)
            if startswith(line, parameter)
                # Extract the parameter value
                value = split(line, " = ")[2]
                break
            end
        end
    end
    return value
end

# Parse parameter value that is an array of float64
function parse_float64_array(array_string::String)::Vector{Float64}
    # Remove "Float64[" and "]" from the array string
    #array_string = replace(array_string, r"Float64\[|\]" => "")
    # Evaluate the string as Julia code
    array_expr = Meta.parse(array_string)
    # Convert the expression to a tuple of Float64
    array_tuple = eval(array_expr)
    # Convert the tuple to a vector
    array = collect(array_tuple)
    return array
end

# --------------------------
# Plot functions
# --------------------------

function generate_log_spaced_values(start_val::Float64, end_val::Float64, num_samples::Int)::Vector
    if start_val <= 0
        start_val = 1e-10  # Set a small positive value instead of 0
    end
    log_start = log10(start_val)
    log_end = log10(end_val)
    log_spaced_vals = Float32(10) .^ LinRange(log_start, log_end, num_samples)
    return log_spaced_vals
end

# --------------------------
# State and transition matrix functions
# --------------------------

function find_pos_of_state(states::Vector, state::Tuple{Int,Int,Int})::Int
    # Find the position without using the lookup table
    return findfirst(x -> x == state, states)
end

function pos_of_state(table::Dict{Tuple{Int, Int, Int}, Int}, state::Tuple{Int, Int, Int})::Int
    get(table, state, -1)  
end

function create_lookup_table(states::Vector)::Dict{Tuple{Int, Int, Int}, Int}
    lookup_table = Dict{Tuple{Int, Int, Int}, Int}()

    for (index, state) in enumerate(states)
        lookup_table[state] = index
    end

    return lookup_table
end

function get_transition_matrix_statdist(transition_matrix::SparseMatrixCSC)::Vector{Float32}
    transition_matrix_transp = transpose(transition_matrix)

    decomp, _ = partialschur(transition_matrix_transp, nev=1, which=:LR, tol=1e-15);
    stat_dist = vec(real(decomp.Q))
    stat_dist /= sum(stat_dist)

    return stat_dist
end

end
