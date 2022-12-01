# File to run the optimize F1 for all 25 races

using DataFrames, CSV
using JuMP, Gurobi
using LinearAlgebra, Random, StatsBase, CategoricalArrays
using Distributions
using Combinatorics
using Dates

function visualize_loops(df)
    G = SimpleDiGraph(length(V))
    for i in 1:nrow(df)
        add_edge!(G, df[i, :i], df[i, :j])
    end
    return G
end

races = ["Sakhir", "Jeddah", "Melbourne", "Shanghai", "Baku", "Miami", "Imola", "Monaco", "Barcelona", "Montreal", "Spielberg", "Silverstone", "Budapest", "Spa", "Zandvoort", "Monza", "Singapore", "Suzuka", "Lusail", "Austin", "Mexico City", "Sao Paulo", "Las Vegas","Yas Marina"]
races_circuitref = ["bahrain", "jeddah", "albert_park", "shanghai", "baku", "miami", "imola", "monaco", "catalunya", "villeneuve", "red_bull_ring", "silverstone", "hungaroring", "spa", "zandvoort", "monza", "marina_bay", "suzuka", "losail", "americas", "rodriguez", "interlagos", "las_vegas", "yas_marina"];

println(now())
println("Reading data")
flush(stdout)

#import Data
all_df = CSV.read("circuits.csv", DataFrame)

select!(all_df, "circuitRef", "location", "lat", "lng")

#only keep rows for the races that we care about, i.e. 2024 schedule
df = all_df[in(races_circuitref).(all_df.circuitRef), :];

df = df[indexin(races_circuitref, df.circuitRef),:];

# F1 teams home locations
teams = ["Mercedes", "Ferrari", "Red Bull", "McLaren", "Aston Martin", "AlphaTauri", "Alpine", "Alfa Romeo", "Haas", "Williams"]
teams_home = ["Brackley", "Maranello", "Milton Keynes", "Woking", "Banbury", "Faenza", "Enstone", "Hinwil", "Kannapolis", "Grove"]
# get the lat and long for each team home to 3 decimal places
teams_lat = [51.683, 44.348, 52.062, 51.323, 52.056, 44.800, 51.750, 44.800, 35.233, 51.750]
teams_lng = [-1.283, 10.926, -0.759, -0.558, -1.328, 11.600, -1.200, 8.300, -80.733, -1.200]

# create a dataframe with the teams and their home locations
teams_df = DataFrame(Team = teams, Home = teams_home, lat = teams_lat, lng = teams_lng)

println(now())
println("Data read")
flush(stdout)

using Geodesy

# Gives distance between two circuits in km
function dist(c1, c2, mat)
    circuit1 = mat[c1, :]
    circuit2 = mat[c2, :]
    lat1 = circuit1[:lat]
    lng1 = circuit1[:lng]
    lat2 = circuit2[:lat]
    lng2 = circuit2[:lng]
    return euclidean_distance(LLA(lat1, lng1, 0), LLA(lat2, lng2, 0)) / 1000
end

println(now())
println("Creating distance matrix")
flush(stdout)

# create a distance matrix for each team home Base
dist_matrices = []
for team in eachrow(teams_df)
    df_temp = copy(df)
    team_name = team[:Team]
    team_home = team[:Home]
    team_lat = team[:lat]
    team_lng = team[:lng]
    # add a row at the beginning of df for this team
    df_temp = vcat(DataFrame(circuitRef = team_name, location = team_home, lat = team_lat, lng = team_lng), df_temp)
    # create a distance matrix for this team
    dist_mat = [dist(i, j, df_temp) for i in 1:size(df_temp, 1), j in 1:size(df_temp, 1)]
    push!(dist_matrices, dist_mat)
end

# add all the distance matrices together
dist_matrix = sum(dist_matrices)

# take the average of the distance matrix
distance_matrix = dist_matrix / size(teams_df, 1)

println(now())
println("Distance matrix created")
flush(stdout)

"""
Returns a `DataFrame` with the values of the variables from the JuMP container `var`.
The column names of the `DataFrame` can be specified for the indexing columns in `dim_names`,
and the name of the data value column by a Symbol `value_col` e.g. :Value
"""
function convert_jump_container_to_df(var::JuMP.Containers.DenseAxisArray;
    dim_names::Vector{Symbol}=Vector{Symbol}(),
    value_col::Symbol=:Value)

    if isempty(var)
        return DataFrame()
    end

    if length(dim_names) == 0
        dim_names = [Symbol("dim$i") for i in 1:length(var.axes)]
    end

    if length(dim_names) != length(var.axes)
        throw(ArgumentError("Length of given name list does not fit the number of variable dimensions"))
    end

    tup_dim = (dim_names...,)

    # With a product over all axis sets of size M, form an Mx1 Array of all indices to the JuMP container `var`
    ind = reshape([collect(k[i] for i in 1:length(dim_names)) for k in Base.Iterators.product(var.axes...)],:,1)

    var_val  = value.(var)

    df = DataFrame([merge(NamedTuple{tup_dim}(ind[i]), NamedTuple{(value_col,)}(var_val[(ind[i]...,)...])) for i in 1:length(ind)])

    return df
end;

println(now())
println("Setting up model")
flush(stdout)

races = 25
home = 1
TH = 2
DH = 6
SH = races - 1 - 3 * TH - 2 * DH
subloops = SH + DH + TH

# get first #races circuitRef from initial df
circuitRefs = df[1:races-1, :circuitRef]

# create vector with string "home" and add circuitRefs to it
circuits = ["home"; circuitRefs];

println(now())
println("Creating powerset")
flush(stdout)

V = 1:races
V_0 = 2:races
ind_end = findall(x->x=="yas_marina", df[!, "circuitRef"])[1] + 1
V_1 = setdiff(V_0, [ind_end])

all_S = collect(Combinatorics.powerset(V_1, 2, length(V) - 2));

println(now())
println("Powerset created")
flush(stdout)


# load vrp results for alpha = 0.15
df_0 = CSV.read("/home/gbonheur/result_vrp_attend_loss/result_vrp_0.15.csv", DataFrame)

# begin TSP

println(now())
println("Beginning TSP calculation")
flush(stdout)

function find_header(df, i)
    nodes = []
    j = df[i, :j]
    if j == home
        return
    end
    push!(nodes, j)
    next = find_header(df, findfirst(df.i .== j))
    # if next is not nothing
    if next !== nothing
        # flatten the array and push
        push!(nodes, next...)
    end
    return nodes
end

# extract double and triple headers from df_manual
sequences = Vector{Vector{Int}}()
for i in 1:nrow(df_0)
    if df_0[i, :i] == home
        push!(sequences, find_header(df_0, i))
    end
end

# get rid of single length sequences
sequences = sequences[map(length, sequences) .> 1]
new = []
for header in sequences
    if length(header) == 3
        # create two arrays of length 2
        # one for the first two nodes
        # one for the last two nodes
        # push them to sequences
        push!(new, header[1:2])
        push!(new, header[2:3])
        # remove the original one
    else
        push!(new, header)
    end
end

sequences = new

emissions_matrix_tsp = zeros(25, 25)

#Iterate through distance matrices

#If the [i,j] is in the sequences (i.e. they are consecutive), only 5 days to get there, automatically impose 5,000 km limit.
#Otherwise, you have at least 12 days to get there, impose 12,000 km limit.

#Driving 

for i in 1:25
    for j in 1:25
        if([i, j] in sequences) #races are consecutive, only 5 days to get there
            if(distance_matrix[i, j] <= 5000)
                emissions_matrix_tsp[i,j]=distance_matrix[i,j]*62 #you can drive, 62 grams/km
            else
                emissions_matrix_tsp[i,j]=distance_matrix[i,j]*500 #you have to fly, 500 grams/km
            end
        else #races are not consecutive, at least 12 days to get there
            if(distance_matrix[i, j] <= 8000)
                emissions_matrix_tsp[i,j]=distance_matrix[i,j]*62 #you can drive, 62 grams/km
            else
                emissions_matrix_tsp[i,j]=distance_matrix[i,j]*500 #you have to fly, 500 grams/km
            end
        end
    end       
end

attendance_loss_matrix = Matrix(CSV.read("attendance_loss_matrix.csv", DataFrame))

# devide attendance loss by 2 for all races not in sequences
for i in 1:25
    for j in 1:25
        if([i, j] in sequences)
            attendance_loss_matrix[i, j] = attendance_loss_matrix[i, j]
        else
            attendance_loss_matrix[i, j] = attendance_loss_matrix[i, j] / 2
        end
    end
end

model2 = Model(Gurobi.Optimizer)
set_optimizer_attribute(model2, "OutputFlag", 1)
set_optimizer_attribute(model2, "Threads", 24)

# variables
@variable(model2, x[i in V, j in V], Bin)

# constraints
# each circuit can only be visited once
@constraint(model2, only_one_in[j in V_0], sum(x[i, j] for i in V) == 1)
@constraint(model2, only_one_out[i in V_0], sum(x[i, j] for j in V) == 1) 

# we cannot go from a circuit to itself
@constraint(model2, no_self_connect[i in V], x[i, i] == 0)

# we must start and end at home, only 1 subloop since we want to cover everything in one go
@constraint(model2, K_petals_in, sum(x[i, 1] for i in V_0) == 1)
@constraint(model2, K_petals_out, sum(x[1, j] for j in V_0) == 1)

# no subloops that do not connect to home
for S in all_S
    @constraint(model2, sum(x[i, j] for i in S, j in S) <= length(S) - 1)
end

# add constraints for double and triple headers
for (i, j) in sequences
    @constraint(model2, x[i, j] == 1)
end

@constraint(model2, x[ind_end, home] == 1)
@constraint(model2, [j in V_1], x[ind_end, j] == 0)

results_tsp = DataFrame(alpha = Float64[], emissions = Float64[], attendance = Float64[])

for alpha in 0 : 0.05 : 1

    println(now())
    println("Running TSP for alpha = $alpha")
    flush(stdout)

    #objective
    @objective(model2, Min, sum(sum(x[i, j] * ((1 - alpha) * emissions_matrix_tsp[i, j] + alpha * 20 * attendance_loss_matrix[i, j]) for i in V) for j in V));

    optimize!(model2)

    println(now())
    println("TSP calculation completed for alpha = $alpha")
    flush(stdout)

    x_opt2 = value.(x);

    df_full_2 = convert_jump_container_to_df(x_opt2, dim_names=[:i, :j], value_col=:x)
    df_2 = df_full_2[df_full_2.x .== 1, :]
    CSV.write("/home/gbonheur/result_tsp_attend/result_tsp_$alpha.csv", df_2)

    tot_em = 0
    tot_attendance = 0
    for row in eachrow(df_2)
        tot_em += emissions_matrix_tsp[row.i, row.j]
        tot_attendance += attendance_loss_matrix[row.i, row.j]
    end
    push!(results_tsp, [alpha, tot_em, tot_attendance])

    file = open("/home/gbonheur/result_tsp_attend/results", "a")
    write(file, "$alpha :\t $tot_em \t $tot_attendance \n")
    close(file)
end

# write results to file
CSV.write("/home/gbonheur/result_tsp_attend/results.csv", results_tsp)