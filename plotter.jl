include("./utils.jl")
include("./strategy.jl")
using LaTeXStrings
using LinearAlgebra
using Statistics
using CairoMakie
using Colors

const line_colors = (RGBA(0.3686, 0.5059, 0.7098),RGBA(0.9216, 0.3843, 0.2078), RGBA(0.5608,0.6902,0.1961), RGBA(0.8784, 0.6118, 0.1412), RGBA(0.5, 0.5, 0.5), RGBA(0.2333, 0.2333, 0.2333))
const markerTypes = (:circle, :rect, :diamond, :cross, :utriangle, :xcross, :vline)
const letter_labels = ["a)", "b)", "c)", "d)", "e)", "f)","g)","h)", "i)","j)","k)","l)"]

# Utility functions

function get_index_k(lis, k) return [res[k] for res in lis] end # aux function to do cross dimentional array indexing

function transform_points(points)
    # transforms points to the new triangle coordinates
    d = π / 3
    tmatrix = [cos(-2d) sin(-2d); cos(-d) sin(-d)]
    tpoints = points * tmatrix
    return tpoints
end
function closest_index(sample_states::Matrix{Float64}, point::Vector{Int64})
    distances = [sqrt((point[1] - state[1])^2 + (point[2] - state[2])^2) for state in eachrow(sample_states)]
    return argmin(distances)
end
function extract_data_vector_sample(all_stts::Matrix, data::Vector, sample_stts::Matrix, sample_size::Int)
    # Initialize dataSample matrix
    dataSample = zeros(length(get_states_strat(sample_size)))
    # Iterate over each point in sample_stts and find its corresponding data value
    for (i, point) in enumerate(eachrow(sample_stts))
        # Find the index of the point in all_stts
        index_in_all_stts = findfirst(all_stt -> all_stt == point, eachrow(all_stts))
        # Copy the data value from data to dataSample
        dataSample[i] = data[index_in_all_stts]
    end
    return dataSample
end
function extract_data_sample(all_stts::Matrix, data::Matrix, sample_stts::Matrix)
    # Initialize dataSample matrix
    dataSample = zeros(size(sample_stts, 1), size(data, 2))
    # Iterate over each point in sample_stts and find its corresponding data value
    for (i, point) in enumerate(eachrow(sample_stts))
        # Find the index of the point in all_stts
        index_in_all_stts = findfirst(all_stt -> all_stt == point, eachrow(all_stts))
        # Copy the data value from data to dataSample
        dataSample[i, :] = data[index_in_all_stts, :]
    end
    return dataSample
end
function rescale_norm(x::Vector{<:Real}, y::Vector{<:Real}, minNorm::Real, maxNorm::Real)
    # Combine x and y into vectors of 2D points
    points = [(x[i], y[i]) for i in 1:length(x)]
    
    # Calculate norms
    norms = [norm(point) for point in points]
    
    # Find min and max norms
    min_norm = minimum(norms)
    max_norm = maximum(norms)
    
    # Rescale norms to range [minNorm, maxNorm]
    scaled_norms = [(norm - min_norm) / (max_norm - min_norm) * (maxNorm - minNorm) + minNorm for norm in norms]
    
    # Rescale vectors
    scaled_points = [(point[1] * (scaled_norms[i] / norms[i]), point[2] * (scaled_norms[i] / norms[i])) for (i, point) in enumerate(points)]
    
    # Separate x and y back
    scaled_x = [point[1] for point in scaled_points]
    scaled_y = [point[2] for point in scaled_points]

    return scaled_x, scaled_y
end

# Plotting functions

function plot_coop_indexes(results,x_axis_scale, subplot_titles, folder_path, filenames::Vector{String}, labelOfFilenames::String) 
    # Make plot folder
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    f = Figure(backgroundcolor = :transparent, size = (1200, 700))

    coop_y_axis = ["Cooperation Index, Iᴴᴴ","Cooperation Index, Iᴴᴬ","Cooperation Index, Iᴬᴴ","Cooperation Index, Iᴴ","Cooperation Index, Iᴴ"]

    for i in range(1,4) # for each type of coop index, we make a plot

        # Define when plot parts appear.
        keepLegend = (i == 4 ? true : false) # Legend = name of each line
        XAxisTitle = (i == 1 || i == 3 ? "Interactions with AA, τ" : "")
        YAxisTitle = coop_y_axis[i]

        ax = Axis(f[i <= 2 ? 1 : 2, i <= 2 ? i : i-2], titlesize=24, xlabelsize=20, ylabelsize=20, 
            xticklabelsize=16, yticklabelsize=16,
            title=subplot_titles[i], xlabel=XAxisTitle, ylabel=YAxisTitle,
            xticks=0:0.1:1.01, yticks=(0:0.2:1.0),xautolimitmargin=(0.0,0.01),
            yautolimitmargin=(0.06,0.06), yminorticksvisible=true)
        ylims!(ax, (-0.05,1.051))

        res = [get_index_k(r, i) for r in results] # get the appropriate coop index from each result

        for v in eachindex(res)
            # Add lines incrementally for each norm
            l = scatterlines!(ax, x_axis_scale, res[v], label=filenames[v], color=line_colors[v],marker=markerTypes[v])
        end

        #if (keepLegend) axislegend(labelOfFilenames,orientation = :horizontal, nbanks=2, position = :lt) end

        text!(ax, 1, 1, text = letter_labels[i], font = :bold, align = (:left, :top), offset = (-430, -2),
            space = :relative, fontsize = 24
        )

        if (i == 4) Legend(f[1, 3], ax, labelOfFilenames) end
    end


    # Save the plot inside the folder
    final_path = joinpath(plot_path, "coop_index.pdf")
    save(final_path, f)
end

function plot_reputations(results, x_axis_scale, line_titles, folder_path, filenames::Vector{String}) 
    # Make plot folder
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    rep_line_colors = (RGB(9/255, 150/255, 11/255), RGB(180/255, 50/255, 25/255), RGB(13/255, 140/255, 227/255),RGB(9/255, 150/255, 11/255), RGB(180/255, 50/255, 25/255), RGB(13/255, 140/255, 227/255),RGB(15/255, 15/255, 15/255))

    f = Figure(backgroundcolor = :transparent, size = (1700, 700))

    for k in range(1, length(results) * 2) # for each SN we make a norm
        i = (k > length(results) ? k - length(results) : k)

        # Define when plot parts appear.
        keepLegend = (k == length(results) || k == length(results)*2 ? true : false) # Legend = name of each line
        XAxisTitle = (k == 1 || k == length(results)+1 ? "Interactions with AA, τ" : "")
        YAxisTitle = (k == 1 ? "Human-Assigned Reputation, rᴴ" : (k == length(results)+1 ? "AA-Assigned Reputation, rᴬ" : ""))

        snlab = k <= length(results) ? filenames[i] : ""

        ax = Axis(f[k <= length(results) ? 1 : 2, k <= length(results) ? k : k-length(results)], titlesize=24, xlabelsize=20, ylabelsize=20, 
            xticklabelsize=16, yticklabelsize=16,
            title=snlab, xlabel=XAxisTitle, ylabel=YAxisTitle,
            xticks=0:0.1:1.01, yticks=(0:0.2:1.0),xautolimitmargin=(0.0,0.01),
            yautolimitmargin=(0.06,0.06), yminorticksvisible=true)

            ylims!(ax, (-0.05,1.051))
            xlims!(ax, (-0.05,1.051))
        
        res = collect(eachrow(reduce(hcat, results[i])))

        for v in eachindex(res)
            if ((k <= length(results) && v in [1,2,3,7]) || (k > length(results) && v in [4,5,6]))
                # Add lines incrementally for each reputation
                scatterlines!(ax, x_axis_scale, res[v], label=line_titles[v], color=rep_line_colors[v],marker=markerTypes[v])
            end
        end

        if (keepLegend) axislegend("Strategy",orientation = :vertical, nbanks=1, position = (1, 0.5)) end

        text!(ax, 1, 1, text = letter_labels[k], font = :bold, align = (:left, :top), offset = (-350, -20),
            space = :relative, fontsize = 24
        )

        #if (k == length(results)) Legend(f[1, length(results) + 1], ax, "Reputation")
        #elseif (k == length(results) * 2) Legend(f[2, length(results) + 1], ax, "Reputation") end
    end

    # Save the plot inside the folder
    final_path = joinpath(plot_path, "reputation.pdf")
    save(final_path, f)
end

function plot_simplex(gradients::Vector, reputation::Vector, cooperation::Vector, stationary::Vector, disagreement::Vector, popsize::Int, sample_size::Int, folder_path::String, path_extension::String, file_extension::String)
    # Make plot folder
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)
    plot_path = joinpath(folder_path, "Plots/Simplex/")
    isdir(plot_path) || mkdir(plot_path)
    plot_path = joinpath(folder_path, "Plots/Simplex/"*path_extension)
    isdir(plot_path) || mkdir(plot_path)

    sample_states = hcat([x[1] for x in get_states_strat(sample_size)], [x[2] for x in get_states_strat(sample_size)])
    sample_states = round.((sample_states .* (popsize / sample_size)))

    # Calculate sum of stationary distributions for the sample states (necessary since sample states < all_states)
    all_states = hcat([x[1] for x in get_states_strat(popsize)], [x[2] for x in get_states_strat(popsize)])
    stationary_sum = zeros(length(get_states_strat(sample_size)))

    for i in 1:length(get_states_strat(popsize))
        closest_indx = closest_index(sample_states, all_states[i, :])
        stationary_sum[closest_indx] += stationary[i]
    end

    # Transform points
    all_transformed_points = transform_points(sample_states)
    gradients_as_points = hcat([x[1] for x in gradients], [x[2] for x in gradients])
    sample_grads = extract_data_sample(all_states, gradients_as_points, sample_states) * sample_size * 1.3
    transformed_gradients = transform_points(sample_grads)
    vector_field = hcat(all_transformed_points, transformed_gradients)

    # only get reputations and coop for specific points
    sample_rep = extract_data_vector_sample(all_states, reputation, sample_states, sample_size)
    sample_coop = extract_data_vector_sample(all_states, cooperation, sample_states, sample_size)
    sample_disagreement = extract_data_vector_sample(all_states, disagreement, sample_states, sample_size)

    # Plotting

    x_axis_lims, y_axis_lims = (-popsize/1.5, popsize/1.5), (-popsize, popsize/4.9)
    
    f_rep = Figure(backgroundcolor = :transparent, size = (800, 700))
    ax_rep = Axis(f_rep[1,1], titlesize=24, title="Average Reputation, rᴴ")

    f_coop = Figure(backgroundcolor = :transparent, size = (800, 700))
    ax_coop = Axis(f_coop[1,1], titlesize=24, title="Cooperation Index, Iᴴᴴ")

    f_disagreement = Figure(backgroundcolor = :transparent, size = (800, 700))
    ax_disagreement = Axis(f_disagreement[1,1], titlesize=24, title="Average Disagreement, qᵈ")

    f_statdist = Figure(backgroundcolor = :transparent, size = (800, 700))
    ax_statdist = Axis(f_statdist[1,1], titlesize=24, title="Stationary Distribution, σₙ")

    f_combined = Figure(backgroundcolor = :transparent, size = (1050, 700))
    ax_combined = Axis(f_combined[1,1], titlesize=24, title="")

    all_axis = [ax_rep, ax_coop, ax_disagreement, ax_statdist]

    
    # add indicators for each pop 
    labelFontSize = 24
    colorbarTickSize = 20
    point_scale = popsize*0.41

    y_axis_increase = popsize/10 # pulls points up
    #annotate!(baseplot,[(0, popsize*0.075, text("Disc", Plots.font("Arial", pointsize=labelFontSize))),(cos(-deg * 2) * popsize * 1.1, sin(-deg * 2) * popsize * 1.1, text("AllC", Plots.font("Arial", pointsize=labelFontSize), halign=:center)),(cos(-deg) * popsize * 1.1, sin(-deg) * popsize * 1.1, text("AllD", Plots.font("Arial", pointsize=labelFontSize), halign=:center))])
    for ax in all_axis
        text!(ax, 1, 1, text = "DISC", font = :bold, align = (:left, :top), offset = (-360, 0), space = :relative, fontsize = labelFontSize)
        text!(ax, 1, 1, text = "ALLC", font = :bold, align = (:left, :top), offset = (-644, -530), space = :relative, fontsize = labelFontSize)
        text!(ax, 1, 1, text = "ALLD", font = :bold, align = (:left, :top), offset = (-70, -530), space = :relative, fontsize = labelFontSize)

        # change plot lims and axis decorations
        xlims!(ax, x_axis_lims)
        ylims!(ax, y_axis_lims)
        hidedecorations!(ax)  
        hidespines!(ax)

        # add black outline
        scatter!(ax, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, marker=:hexagon, markersize=point_scale*1.1, color=:black)
    end
    minNorm, maxNorm = 1, 5
    vector_field[:, 3],vector_field[:, 4] = rescale_norm(vector_field[:, 3],vector_field[:, 4], minNorm, maxNorm)

    #rep = scatter(baseplot,all_transformed_points[:, 1], all_transformed_points[:, 2], markersize=point_scale, zcolor=sample_rep, color=cgrad(:RdYlBu_4, rev = false),markerstrokewidth=0, shape = :h)
    rep_color =:RdBu_9
    scatter!(ax_rep, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, marker=:hexagon, markersize=point_scale, color=sample_rep, colorrange = (0.0, 1.0), colormap=rep_color)
    Colorbar(f_rep[1,2], colormap=rep_color, limits=(0,1), size=15, ticks = 0:0.1:1, ticksize=10,tellheight=true,height = Relative(2.5/4),ticklabelsize=colorbarTickSize, label = "Average Reputation, rᴴ", labelsize=20)
    arrows!(ax_rep, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, vector_field[:, 3],vector_field[:, 4]);

    coop_color = :Purples_5
    scatter!(ax_coop, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, marker=:hexagon, markersize=point_scale, color=sample_coop, colorrange = (0.0, 1.0), colormap=coop_color)
    Colorbar(f_coop[1,2], colormap=coop_color, limits=(0,1), size=15, ticks = 0:0.1:1, ticksize=10,tellheight=true,height = Relative(2.5/4),ticklabelsize=colorbarTickSize, label = "Cooperation Index, Iᴴᴴ", labelsize=20)
    arrows!(ax_coop, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, vector_field[:, 3],vector_field[:, 4]);

    disagreement_color = Reverse(:RdYlGn_9)
    scatter!(ax_disagreement, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, marker=:hexagon, markersize=point_scale, color=sample_disagreement, colorrange = (0.0, 0.5), colormap=disagreement_color)
    Colorbar(f_disagreement[1,2], colormap=disagreement_color, limits=(0,0.5), size=15, ticks = 0:0.1:0.5, ticksize=10,tellheight=true,height = Relative(2.5/4),ticklabelsize=colorbarTickSize, label = "Average Disagreement, qᵈ", labelsize=20)
    arrows!(ax_disagreement, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, vector_field[:, 3],vector_field[:, 4]);

    normalized_stationary = stationary_sum ./ maximum(stationary_sum)
    statdist_color = :matter
    scatter!(ax_statdist, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, marker=:hexagon, markersize=point_scale, color=normalized_stationary, colorrange = (0.0, 1.0), colormap=statdist_color)
    Colorbar(f_statdist[1,2], colormap=statdist_color, limits=(0,1), size=15, ticks = 0:0.1:1, ticksize=10,tellheight=true,height = Relative(2.5/4),ticklabelsize=colorbarTickSize, label = "Stationary Distribution, σₙ", labelsize=20)
    arrows!(ax_statdist, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, vector_field[:, 3],vector_field[:, 4]);

    # Combined plot
    # Repeat the same stuff for the stat dist but add the reputations in the corner
    combinedpoint_scale = popsize*0.46
    minipoint_scale = popsize*0.16
    
    scatter!(ax_combined, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, marker=:hexagon, markersize=combinedpoint_scale*1.1, color=:black)
    scatter!(ax_combined, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, marker=:hexagon, markersize=combinedpoint_scale, color=normalized_stationary, colorrange = (0.0, 1.0), colormap=statdist_color)

    scatter!(ax_combined, (all_transformed_points[:, 1]./3).-popsize/2.5, (all_transformed_points[:, 2]./3).+popsize/8, marker=:hexagon, markersize=minipoint_scale*1.25, color=:black)
    scatter!(ax_combined, (all_transformed_points[:, 1]./3).-popsize/2.5, (all_transformed_points[:, 2]./3).+popsize/8, marker=:hexagon, markersize=minipoint_scale, color=sample_rep, colorrange = (0.0, 1.0), colormap=rep_color)

    scatter!(ax_combined, (all_transformed_points[:, 1]./3).+popsize/2.5, (all_transformed_points[:, 2]./3).+popsize/8, marker=:hexagon, markersize=minipoint_scale*1.25, color=:black)
    scatter!(ax_combined, (all_transformed_points[:, 1]./3).+popsize/2.5, (all_transformed_points[:, 2]./3).+popsize/8, marker=:hexagon, markersize=minipoint_scale, color=sample_disagreement, colorrange = (0.0, 0.5), colormap=disagreement_color)

    Colorbar(f_combined[1,2], colormap=statdist_color, limits=(0,1), size=15, ticks = 0:0.1:1, ticksize=10,tellheight=true,height = Relative(2.5/4),ticklabelsize=colorbarTickSize, label = "Stationary Distribution, σₙ", labelsize=20)
    Colorbar(f_combined[1,3], colormap=rep_color, limits=(0,1), size=15, ticks = 0:0.1:1, ticksize=10,tellheight=true,height = Relative(2.5/4),ticklabelsize=colorbarTickSize, label = "Average Reputation, rᴴ", labelsize=20)
    Colorbar(f_combined[1,4], colormap=disagreement_color, limits=(0,0.5), size=15, ticks = 0:0.1:0.5, ticksize=10,tellheight=true,height = Relative(2.5/4),ticklabelsize=colorbarTickSize, label = "Average Disagreement, qᵈ", labelsize=20)

    arrows!(ax_combined, all_transformed_points[:, 1], all_transformed_points[:, 2].+y_axis_increase, vector_field[:, 3],vector_field[:, 4]);

    text!(ax_combined, 1, 1, text = "DISC", font = :bold, align = (:left, :top), offset = (-383, -8), space = :relative, fontsize = labelFontSize)
    text!(ax_combined, 1, 1, text = "ALLC", font = :bold, align = (:left, :top), offset = (-705, -550), space = :relative, fontsize = labelFontSize)
    text!(ax_combined, 1, 1, text = "ALLD", font = :bold, align = (:left, :top), offset = (-69, -550), space = :relative, fontsize = labelFontSize)
    # change plot lims and axis decorations
    xlims!(ax_combined, x_axis_lims)
    ylims!(ax_combined, y_axis_lims)
    hidedecorations!(ax_combined)  
    hidespines!(ax_combined)

    # Save all plots
    save(joinpath(plot_path, "simplex_rep_"*file_extension*".pdf"), f_rep)  
    save(joinpath(plot_path, "simplex_coop_"*file_extension*".pdf"), f_coop)  
    save(joinpath(plot_path, "simplex_disagreement_"*file_extension*".pdf"), f_disagreement)  
    save(joinpath(plot_path, "simplex_statdist_"*file_extension*".pdf"), f_statdist)
    save(joinpath(plot_path, "_simplex_combined_"*file_extension*".pdf"), f_combined)    
end

function plot_disagreement(results,x_axis_scale, folder_path, filenames::Vector{String}, labelOfFilenames::String)
    # Plot difference between ALLC and ALLD rep, and DISC and ALLD rep.
    # Make plot folder
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    f = Figure(backgroundcolor = :transparent, size = (600, 350))

    XAxisTitle = "Interactions with AA, τ"
    YAxisTitle = "Average Disagreement, qᵈ"

    ax = Axis(f[1,1], titlesize=24, xlabelsize=20, ylabelsize=20, 
        xticklabelsize=16, yticklabelsize=16,
        title="", xlabel=XAxisTitle, ylabel=YAxisTitle,
        xticks=0:0.1:1.01, yticks=(0:0.1:0.5),xautolimitmargin=(0.0,0.01),
        yautolimitmargin=(0.06,0.06), yminorticksvisible=true)
    ylims!(ax, (-0.05,0.526))

    for v in eachindex(results)
        # Add lines incrementally for each norm
        scatterlines!(ax, x_axis_scale, results[v], label=filenames[v], color=line_colors[v],marker=markerTypes[v])
    end

    Legend(f[1, 2], ax, labelOfFilenames)

    # Save the plot inside the folder
    final_path = joinpath(plot_path, "disagreement.pdf")
    save(final_path, f)
end

function plot_diff_reps(results,x_axis_scale, folder_path, filenames::Vector{String}, labelOfFilenames::String, filetag::String)
    # Make plot folder
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    f = Figure(backgroundcolor = :transparent, size = (600, 350))

    XAxisTitle = "Interactions with AA, τ"
    YAxisTitle = "Reputation Difference"

    ax = Axis(f[1,1], titlesize=24, xlabelsize=20, ylabelsize=20, 
        xticklabelsize=16, yticklabelsize=16,
        title="", xlabel=XAxisTitle, ylabel=YAxisTitle,
        xticks=0:0.1:1.01, yticks=(-1.0:0.2:1.0),xautolimitmargin=(0.0,0.01),
        yautolimitmargin=(0.06,0.06), yminorticksvisible=true)
    ylims!(ax, (-1.05,1.051))

    for v in eachindex(results)
        # Add lines incrementally for each norm. Line with 
        scatterlines!(ax, x_axis_scale, results[v], label=filenames[v], color=line_colors[v],marker=markerTypes[v])
    end

    Legend(f[1, 2], ax, labelOfFilenames)

    # Save the plot inside the folder
    final_path = joinpath(plot_path, "reputation_difference_$filetag.pdf")
    save(final_path, f)
end

function plot_grad(AA_strat::Int, norms::Vector, vectorNAAs::Vector, popsize::Int, execError::Float64, b::Float64, gossipRounds::Int, mutChance::Float64, imitAAs::Bool, folder_path::String, filenames::Vector{String}) 
    # Make plot folder
    isdir(folder_path) || mkdir(folder_path)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    f = Figure(backgroundcolor = :transparent, size = (1800, 350))
    axesList = []

    maxRes,minRes = -Inf, Inf

    rangeAllD = 0:1:popsize
    rangefAllD = [a/popsize for a in rangeAllD] 

    lineFadeFactor = 4.5

    for i in eachindex(norms)

        ylab = (i == 1 ? "Gradient of Selection AllD -> Disc" : "")
        keepLegend = (i == length(norms) ? true : false)

        ax = Axis(f[1, i], titlesize=24, xlabelsize=20, ylabelsize=20, 
            xticklabelsize=16, yticklabelsize=16,
            title=filenames[i], xlabel="Fraction of Disc", ylabel=ylab,xautolimitmargin=(0.00,0.01),
            xticks=0:0.1:1.01,
            yautolimitmargin=(0.06,0.06), yminorticksvisible=true)

        lines!(ax, rangefAllD, [0 for var in eachindex(rangeAllD)], color=:black)
    
        # create each line and add it to the plot
        for fAAs in vectorNAAs

            results = [grad_disc_alld(0, nAllD, norms[i], fAAs, AA_strat, popsize, execError, b, gossipRounds, mutChance,imitAAs) for nAllD in rangeAllD]

            maxRes,minRes = max(maximum(results),maxRes), min(minimum(results), minRes)

            lines!(ax, rangefAllD, results, color=(line_colors[i],(1-fAAs*lineFadeFactor)),label="A = "*string(fAAs))

            # Check for intersections with x-axis
            x_points::Vector{Float64} = []
            for t in 1:length(results)-1
                if sign(results[t]) * sign(results[t+1]) < 0
                    # Interpolate to find the precise intersection point
                    m = (results[t+1] - results[t]) / (rangefAllD[t+1] - rangefAllD[t])
                    c = results[t] - m * rangefAllD[t]
                    x_intersection = -c / m
                    
                    push!(x_points, x_intersection)
                end
            end
            scatter!(x_points, zeros(length(x_points)), color=line_colors[i], strokewidth = 1, strokecolor = :black)

        end
        if (keepLegend) axislegend("Inter. AA", position = :rb) end

        text!(ax, 1, 1, text = letter_labels[i], font = :bold, align = (:left, :top), offset = (-275, -2), space = :relative, fontsize = 24)

        push!(axesList, ax)
    end

    # Change the axis limits across all axis
    minRes = round(minRes, sigdigits=2)-0.02
    maxRes = round(maxRes, sigdigits=4)+0.02
    minRes -= (isodd(Int(round((minRes*100) % 10))) ? 0.01 : 0.0 )
    maxRes +=(isodd(Int(round((minRes*100) % 10))) ? 0.01 : 0.0 )

    for ax in axesList
        ax.yticks=minRes:0.04:maxRes
        ylims!(ax, minRes, maxRes)
    end

    # Save the plot inside the folder
    plot_path = joinpath(plot_path, "grad_sel_alld_disc.pdf")
    save(plot_path, f)

    Memoization.empty_all_caches!()
end

function b_study_plot(folder_path::String, filenames::Vector{String}, labelOfFilenames::String)
    # Make plot folder
    isdir(folder_path) || mkdir(folder_path)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    res_path::String = joinpath(folder_path, "Results/ResultsBackup")

    b_vals::Vector = utils.parse_float64_array(String(utils.get_parameter_value(folder_path,"Range of Values")))

    results = []
    for name in filenames
        push!(results, utils.deserialize_file(joinpath(res_path,name*"_results.jls")))
    end

    results_coop = []
    for i in eachindex(filenames)
        push!(results_coop, [results[i][k][1][1][1] for k in eachindex(b_vals)])
    end

    # Create a single plot for each AAsPop
    f = Figure(backgroundcolor = :transparent, size = (600, 350))

    ax = Axis(f[1, 1], titlesize=24, xlabelsize=20, ylabelsize=20, 
            xticklabelsize=16, yticklabelsize=16,
            title="", xlabel="Benefit-to-cost ratio, b/c", ylabel=ylabel="Cooperation Index, Iᴴᴴ",xautolimitmargin=(0.00,0.01),
            xticks=1:1:maximum(b_vals)+0.1, yticks=0.0:0.1:1.01,
            yautolimitmargin=(0.06,0.06))

    for i in eachindex(results_coop)
        # Add lines incrementally for each norm
        scatterlines!(ax, b_vals, results_coop[i], label=filenames[i], color=line_colors[i],marker=markerTypes[i])
    end 
    ylims!(ax, 0, 1)

    Legend(f[1,2], ax, labelOfFilenames)
    # Save the plot inside the folder
    plot_path = joinpath(plot_path, "b_study.pdf")
    save(plot_path, f)
end

function b_study_multiple_plot(folder_path::String, all_filenames::Vector, labelOfFilenames::String, filetags::Vector{String}=[], AA_fracs::Vector{Float64}=[])
    # Make plot folder
    isdir(folder_path) || mkdir(folder_path)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    res_path::String = joinpath(folder_path, "Results/ResultsBackup")

    b_vals::Vector = utils.parse_float64_array(String(utils.get_parameter_value(folder_path,"Range of Values")))

    # Create a single plot for all results - Cooperation
    f = Figure(backgroundcolor = :transparent, size = (850, 650))

    for t in eachindex(all_filenames)
        results = []
        filenames = all_filenames[t]
        for name in filenames
            push!(results, utils.deserialize_file(joinpath(res_path,name*"_results.jls")))
        end
        
        results_coop = []
        for i in eachindex(filenames)
            push!(results_coop, [results[i][k][1][1][1] for k in eachindex(b_vals)])
        end
        
        xlabel = t == 1 ? "Benefit-to-cost ratio, b/c" : ""
        ylabel = t == 1  || t == 3 ? "Cooperation Index, Iᴴᴴ" : ""
        
        ypos = (t < 3 ? 1 : 2)
        xpos = (t < 3 ? t : t - 2)
        ax = Axis(f[ypos, xpos], titlesize=24, xlabelsize=20, ylabelsize=20, 
        xticklabelsize=16, yticklabelsize=16,
        title=filetags[t], xlabel=xlabel, ylabel=ylabel,xautolimitmargin=(0.00,0.01),
        xticks=1:1:maximum(b_vals)+0.1, yticks=0.0:0.1:1.01,
        yautolimitmargin=(0.06,0.06))
        
        for i in eachindex(results_coop)
            # Add lines incrementally for each norm
            scatterlines!(ax, b_vals, results_coop[i], label=string(AA_fracs[i]), color=(line_colors[t],(1-i/(length(results_coop)*1.5))),marker=markerTypes[i])
        end 
        ylims!(ax, 0, 1)

    end
    elements = []
    labels = [string(frac) for frac in AA_fracs]

    for i in eachindex(all_filenames[1])
        push!(elements,
        [LineElement(color = (RGBA(0.5,0.5,0.5),(1-i/(length(all_filenames[1])*1.5))), linestyle = nothing),
        MarkerElement(color = (RGBA(0.5,0.5,0.5),(1-i/(length(all_filenames[1])*1.5))), marker = markerTypes[i])])
    end
    Legend(f[1,3], elements, labels, labelOfFilenames)

    # Save the plot inside the folder
    plot_path_fin = joinpath(plot_path, "b_study_all_cooperation.pdf")
    save(plot_path_fin, f)

    # Create a single plot for all results -- disagreement
    f = Figure(backgroundcolor = :transparent, size = (850, 650))

    for t in eachindex(all_filenames)
        results = []
        filenames = all_filenames[t]
        for name in filenames
            push!(results, utils.deserialize_file(joinpath(res_path,name*"_results.jls")))
        end
        
        results_coop = []
        for i in eachindex(filenames)
            push!(results_coop, [results[i][k][5][1][2] for k in eachindex(b_vals)])
        end
        
        xlabel = t == 1 ? "Benefit-to-cost ratio, b/c" : ""
        ylabel = t == 1 || t == 3 ? "Average Disagreement, qᵈ" : ""
        
        ypos = (t < 3 ? 1 : 2)
        xpos = (t < 3 ? t : t - 2)
        ax = Axis(f[ypos, xpos], titlesize=24, xlabelsize=20, ylabelsize=20, 
        xticklabelsize=16, yticklabelsize=16,
        title=filetags[t], xlabel=xlabel, ylabel=ylabel,xautolimitmargin=(0.00,0.01),
        xticks=1:1:maximum(b_vals)+0.1, yticks=0.0:0.1:0.59,
        yautolimitmargin=(0.06,0.06))
        
        for i in eachindex(results_coop)
            # Add lines incrementally for each norm
            scatterlines!(ax, b_vals, results_coop[i], label=string(AA_fracs[i]), color=(line_colors[t],(1-i/(length(results_coop)*1.5))),marker=markerTypes[i])
        end 

        ylims!(ax, -0.02, 0.51)
    end
    elements = []
    labels = [string(frac) for frac in AA_fracs]
    
    for i in eachindex(all_filenames[1])
        push!(elements,
        [LineElement(color = (RGBA(0.5,0.5,0.5),(1-i/(length(all_filenames[1])*1.5))), linestyle = nothing),
        MarkerElement(color = (RGBA(0.5,0.5,0.5),(1-i/(length(all_filenames[1])*1.5))), marker = markerTypes[i])])
    end
    Legend(f[1,3], elements, labels, labelOfFilenames)
    
    # Save the plot inside the folder
    plot_path_fin = joinpath(plot_path, "b_study_all_disagreement.pdf")
    save(plot_path_fin, f)
end

function error_study_plot(folder_path::String, taus_to_test::Vector, error_values::Vector,all_filenames::Vector, filetags::Vector{String},labelOfFilenames::String)

    # Make plot folder
    isdir(folder_path) || mkdir(folder_path)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    res_path::String = joinpath(folder_path, "Results/ResultsBackup")

    indexRes1_values = [1, 5]
    indexRes2_values = [1, 2]
    ylabel = ["Cooperation Index, Iᴴᴴ","Disagreement, qᵈ"]
    titletag = ["Cooperation", "Disagreement"]
    finaltag = ["coop","disagreement"]
    yticks = [0.0:0.2:1.01, 0.0:0.1:0.51]

    # Create a plot for both execution and assessment error for both cooperation and disagreement
    
    for t in eachindex(all_filenames)   # each filename is a pair [execFilenames, assessFilenames]

        for iter in [1,2]   # 1 for coop, 2 for disagreement
            f = Figure(backgroundcolor = :transparent, size = (1150, 350))

            # First do execution error
            results = []
            filenames = all_filenames[t][1]
            for name in filenames
                push!(results, utils.deserialize_file(joinpath(res_path,name*"_results.jls")))
            end
            
            results_coop = []
            for i in eachindex(filenames)
                push!(results_coop, [results[i][k][indexRes1_values[iter]][1][indexRes2_values[2]] for k in eachindex(error_values)])
            end

            ax = Axis(f[1, 1], titlesize=24, xlabelsize=20, ylabelsize=20, 
            xticklabelsize=16, yticklabelsize=16,
            title=titletag[iter]*" vs Execution Error", xlabel="Execution error, eₑ", ylabel=ylabel[iter],xautolimitmargin=(0.00,0.01),
            xticks=[1e-4, 1e-3, 1e-2, 0.1, 0.5], yticks=yticks[iter], 
            yautolimitmargin=(0.06,0.06),xscale=log10,xtickformat = x -> ["1e-4", "1e-3", "1e-2", "0.1", "0.5"])
            
            for i in eachindex(results_coop)
                # Add lines incrementally for each norm
                scatterlines!(ax, error_values, results_coop[i], label=string(taus_to_test[i]), color=(line_colors[t],(1-i/(length(results_coop)*1.5))),marker=markerTypes[i])
            end 
            if (iter == 1)
                ylims!(ax, 0, 1)
            else
                ylims!(ax, -0.02, 0.52)
            end
            xlims!(ax,1e-4, 0.5)

            # then assessment error
            results = []
            filenames = all_filenames[t][2]
            for name in filenames
                push!(results, utils.deserialize_file(joinpath(res_path,name*"_results.jls")))
            end
            
            results_coop = []
            for i in eachindex(filenames)
                push!(results_coop, [results[i][k][indexRes1_values[iter]][1][indexRes2_values[2]] for k in eachindex(error_values)])
            end

            ax = Axis(f[1, 2], titlesize=24, xlabelsize=20, ylabelsize=20, 
            xticklabelsize=16, yticklabelsize=16,
            title=titletag[iter]*" vs Assessment Error", xlabel="Assessment error, eₐ", ylabel="",xautolimitmargin=(0.00,0.01),
            xticks=[1e-4, 1e-3, 1e-2, 0.1, 0.5], yticks=yticks[iter],
            yautolimitmargin=(0.06,0.06),xscale=log10,xtickformat = x -> ["1e-4", "1e-3", "1e-2", "0.1", "0.5"])
            
            for i in eachindex(results_coop)
                # Add lines incrementally for each norm
                scatterlines!(ax, error_values, results_coop[i], label=string(taus_to_test[i]), color=(line_colors[t],(1-i/(length(results_coop)*1.5))),marker=markerTypes[i])
            end 
            if (iter == 1)
                ylims!(ax, 0, 1)
            else
                ylims!(ax, -0.02, 0.52)
            end
            xlims!(ax,1e-4, 0.5)

            # Make legend

            elements = []
            labels = [string(frac) for frac in taus_to_test]
        
            for i in eachindex(filenames)
                push!(elements,
                [LineElement(color = (line_colors[t],(1-i/(length(results_coop)*1.5))), linestyle = nothing),
                MarkerElement(color = (line_colors[t],(1-i/(length(results_coop)*1.5))), marker = markerTypes[i])])
            end
            Legend(f[1,3], elements, labels, labelOfFilenames)

            # Add title label with SN tag
            Label(f[0, 1:2], filetags[t], fontsize=24, halign = :center,font = :bold)
        
            # Save the plot inside the folder
            plot_path_fin = joinpath(plot_path, filetags[t]*"_"*finaltag[iter]*"_error_study.pdf")
            save(plot_path_fin, f)
        end
    end
end

function intermediate_norms_plot(folder_path::String, filename::String, title::String, filetag::String, norms::Vector, simplify::Bool=false)
    # Make plot folder
    isdir(folder_path) || mkdir(folder_path)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    # Make cooperation and disagreement folder
    coop_path = joinpath(plot_path, "Cooperation")
    isdir(coop_path) || mkdir(coop_path)
    disagreement_path = joinpath(plot_path, "Disagreement")
    isdir(disagreement_path) || mkdir(disagreement_path)

    res_path::String = joinpath(folder_path, "Results/ResultsBackup")

    results = utils.deserialize_file(joinpath(res_path,filename*"_results.jls"))
    x_range = 0:0.1:1.0

    for type in [1,2]   # type 1 = coop, type 2 = disagreement
        
        res = [results[k][1][1][1] for k in eachindex(norms)]
        if type == 2
            res = [results[k][5][1][2] for k in eachindex(norms)]
        end
        results_matrix = reshape(res, length(x_range), length(x_range))
        
        label = (type == 1 ? "Cooperation Index, Iᴴᴴ" : "Disagreement, qᵈ")
        crange = (type == 1 ? (0,1) : (0, 0.5))
        # Plot the heatmap
        f = Figure(backgroundcolor = :transparent, size = (600, 600))
        ax = Axis(f[1, 1], xlabel="α¹", ylabel="α²", title=title)
        heatmap!(ax, x_range, x_range, results_matrix, colorrange=crange, colormap=:viridis)
        Colorbar(f[1, 2], label=label, limits=crange)

        if (!simplify)

            # Add value labels on top of each cell
            for i in eachindex(x_range), j in eachindex(x_range)
                txtcolor = results_matrix[i, j] < 0.3 / type ? :white : :black
                text!(ax, "$(round(results_matrix[i,j], digits = 2))", position = (x_range[i], x_range[j]),
                    color = txtcolor, align = (:center, :center), fontsize=6)
            end
            text!(ax, "SH", position=(0.04, 0.04), color=:white)
            text!(ax, "IS", position=(0.945, 0.045), color=:white)
            text!(ax, "SJ", position=(0.04, 0.945), color=:white)
            text!(ax, "SS", position=(0.94, 0.94), color=:white)
        else
            text!(ax, "Shunning", fontsize=22, position=(0.01, 0.0), color=:white)
            text!(ax, "Image\nScore", fontsize=22, position=(0.84, -0.02), color=:white)
            text!(ax, "Stern-Judging", fontsize=22, position=(0.01, 0.96), color=:white)
            text!(ax, "Simple\nStanding", fontsize=22, position=(0.80, 0.92), color=:white)
        end


        # Save the plot inside the folder
        tag = (type == 1 ? "_coop" : "_disagreement")
        final_path = (type == 1 ? coop_path : disagreement_path)
        plot_path = joinpath(final_path, filetag*tag*".pdf")
        save(plot_path, f)
    end
end

function plot_coop_indexes_public_vs_private(folder_path::String, folder_path_public::String, folder_path_private::String, filenames::Vector{String}, filename_reputation::String, bc_val::Float64)
    # Function used purely for Figure 2 of the main text, showcasing the difference between public and private reputations
    # Requires the folder where the plot is stored (folder_path), the folders of the results for public and private reputations, 
    # the filenames associated with each results, the filename whose reputation is shown on the right side, and the bc value to evaluate in the right and plot a line in on the left
    
    # Make plot folder
    isdir(folder_path) || mkdir(folder_path)
    plot_path = joinpath(folder_path, "Plots")
    isdir(plot_path) || mkdir(plot_path)

    res_path_pub::String = joinpath(folder_path_public, "Results/ResultsBackup")
    res_path_priv::String = joinpath(folder_path_private, "Results/ResultsBackup")

    b_vals::Vector = utils.parse_float64_array(String(utils.get_parameter_value(folder_path_public,"Range of Values")))

    results_pub = []
    for name in filenames
        push!(results_pub, utils.deserialize_file(joinpath(res_path_pub,name*"_results.jls")))
    end

    results_priv = []
    for name in filenames
        push!(results_priv, utils.deserialize_file(joinpath(res_path_priv,name*"_results.jls")))
    end

    results_coop_pub = []
    for i in eachindex(filenames)
        push!(results_coop_pub, [results_pub[i][k][1][1][1] for k in eachindex(b_vals)])
    end

    results_coop_priv = []
    for i in eachindex(filenames)
        push!(results_coop_priv, [results_priv[i][k][1][1][1] for k in eachindex(b_vals)])
    end

    # Create a single plot for each AAsPop
    f = Figure(backgroundcolor = :transparent, size = (600, 350))

    ax = Axis(f[1:2, 1], titlesize=24, xlabelsize=20, ylabelsize=20, 
            xticklabelsize=16, yticklabelsize=16,
            title="", xlabel="Benefit-to-cost ratio, b/c", ylabel="Cooperation Index, Iᴴᴴ",xautolimitmargin=(0.00,0.01),
            xticks=1:1:maximum(b_vals)+0.1, yticks=0.0:0.1:1.01,
            yautolimitmargin=(0.06,0.06))

    for i in eachindex(results_coop_pub)
        # Add lines incrementally for each norm in public reps
        scatterlines!(ax, b_vals, results_coop_pub[i], color=(line_colors[i], 0.4),marker=markerTypes[i])
    end 
    for i in eachindex(results_coop_priv)
        # Add lines incrementally for each norm in private reps
        scatterlines!(ax, b_vals, results_coop_priv[i], label=filenames[i], color=line_colors[i],marker=markerTypes[i])
    end 
    vlines!(ax, bc_val, color=:black, linestyle=:dash, linewidth=1)
    ylims!(ax, 0, 1)

    colsize!(f.layout, 1, Relative(0.75))
    axislegend(ax, merge = true, unique = true, labelsize=10, titlesize=12, position = :rb)

    # Bar plots with average reputation
    
    #Find index of results closest to b/c=3
    index_of_bc = argmin(abs.(b_vals .- bc_val))
    index_of_norm = findfirst(==(filename_reputation), filenames)

    result_rep_pub = results_pub[index_of_norm][index_of_bc][2][1]
    results_rep_priv = results_priv[index_of_norm][index_of_bc][2][1]

    strats = ["ALLC","ALLD","DISC"]

    ax = Axis(f[1, 2], titlesize=12, xlabelsize=14, ylabelsize=14, 
    xticklabelsize=12, yticklabelsize=12,
    title=filename_reputation*" (Public)", xlabel="", ylabel=ylabel="Average Reputation, rᴴ",xticks=(1:3, strats), yticks=0.0:0.5:1.01,yautolimitmargin=(0.06,0.06))

    barplot!(ax, 1:3, result_rep_pub[1:3], color=line_colors[index_of_norm])
    ylims!(ax, 0, 1)

    ax = Axis(f[2, 2], titlesize=12, xlabelsize=14, ylabelsize=14, 
    xticklabelsize=12, yticklabelsize=12,
    title=filename_reputation*" (Private)", xlabel="", ylabel=ylabel="",xticks=(1:3, strats), yticks=0.0:0.5:1.01,yautolimitmargin=(0.06,0.06))

    barplot!(ax, 1:3, results_rep_priv[1:3], color=line_colors[index_of_norm])
    ylims!(ax, 0, 1)

    # Save the plot inside the folder
    plot_path = joinpath(plot_path, "b_study.pdf")
    save(plot_path, f)
end

function run_all_plots(folder_path::String, filenames::Vector{String}, sn_labels::Vector{String}, labelOfFilenames::String, includeSimplex::Bool=true)
    
    res_path::String = joinpath(folder_path, "Results/ResultsBackup")
    
    # Change what part of the results structure we want to extract
    # for now, a plot with all the coop indexes
    interaction_AA_vals::Vector = utils.parse_float64_array(String(utils.get_parameter_value(folder_path,"Range of Values")))
    pops = parse(Int,utils.get_parameter_value(folder_path,"Population Size"))

    results = []
    for name in filenames
        push!(results, utils.deserialize_file(joinpath(res_path,name*"_results.jls")))
    end

    titles_coop::Vector = ["Human-Human", "Human-AA", "AA-Human", "Human", "Collective"]
    
    results_coop = []
    for i in eachindex(filenames)
        push!(results_coop, [results[i][k][1][1] for k in eachindex(interaction_AA_vals)])
    end

    titles_rep::Vector = ["ALLC", "ALLD", "DISC", "ALLC", "ALLD", "DISC", "AA"]
    
    results_rep = []
    for i in eachindex(filenames)
        push!(results_rep, [results[i][k][2][1] for k in eachindex(interaction_AA_vals)])
    end
    pop!(results_rep)   # Take AG and AB
    pop!(results_rep)

    results_disagreement = []
    for i in eachindex(filenames)
        push!(results_disagreement, [results[i][k][5][1][2] for k in eachindex(interaction_AA_vals)])
    end

    results_rep_diff_allc_H, results_rep_diff_disc_H, results_rep_diff_allc_A, results_rep_diff_disc_A = [],[],[],[]
    for i in 1:(length(filenames)-2)
        push!(results_rep_diff_allc_H, [results[i][k][2][1][1] - results[i][k][2][1][2] for k in eachindex(interaction_AA_vals)])  # allC-allD - Humans
        push!(results_rep_diff_disc_H, [results[i][k][2][1][3] - results[i][k][2][1][2] for k in eachindex(interaction_AA_vals)])  # disc-allD
        push!(results_rep_diff_allc_A, [results[i][k][2][1][4] - results[i][k][2][1][5] for k in eachindex(interaction_AA_vals)])  # allC-allD - AA
        push!(results_rep_diff_disc_A, [results[i][k][2][1][6] - results[i][k][2][1][5] for k in eachindex(interaction_AA_vals)])  # disc-allD
    end
    plot_coop_indexes(results_coop, interaction_AA_vals, titles_coop, folder_path, sn_labels,labelOfFilenames)
    plot_reputations(results_rep, interaction_AA_vals, titles_rep, folder_path, sn_labels)
    plot_disagreement(results_disagreement, interaction_AA_vals, folder_path, sn_labels, labelOfFilenames)
    plot_diff_reps(results_rep_diff_allc_H, interaction_AA_vals, folder_path, sn_labels, labelOfFilenames, "allc_H")
    plot_diff_reps(results_rep_diff_disc_H, interaction_AA_vals, folder_path, sn_labels, labelOfFilenames, "disc_H")
    plot_diff_reps(results_rep_diff_allc_A, interaction_AA_vals, folder_path, sn_labels, labelOfFilenames, "allc_A")
    plot_diff_reps(results_rep_diff_disc_A, interaction_AA_vals, folder_path, sn_labels, labelOfFilenames, "disc_A")

    # Make simples with all the social norms, for a given number of AAs
    sample_size = 20 # how many points to include in each simplex side
    if (includeSimplex)
        for interVal in 0:0.1:1.0
            interAA_index = findfirst(x -> x == interVal, interaction_AA_vals)

            for i in eachindex(sn_labels)
                gradients_i::Vector = results[i][interAA_index][6]
                reputation_i::Vector = utils.get_average_rep_states(results[i][interAA_index], pops)
                cooperation_i::Vector = get_index_k(results[i][interAA_index][1][2], 1)
                stationary_i::Vector = results[i][interAA_index][3]
                disagreement_i = get_index_k(results[i][interAA_index][5][2], 2)
                plot_simplex(gradients_i, reputation_i, cooperation_i, stationary_i, disagreement_i, pops, sample_size, folder_path, "AAs"*string(interVal), sn_labels[i]*"_"*string(interVal))
            end
        end
    end
end

#plot_coop_indexes_public_vs_private("Results/b_study_pub_priv_justIS_SJ","Results/b-study-all-gossip","Results/b-study-baseline", ["Image Score", "Stern-Judging"], "Stern-Judging", 3.0)
