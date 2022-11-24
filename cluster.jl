# File to run the optimize F1 for all 25 races

using DataFrames, CSV
using JuMP, Gurobi
using LinearAlgebra, Random, Printf, StatsBase, CategoricalArrays
using Distributions
using Combinatorics

function visualize_loops(df)
    G = SimpleDiGraph(length(V))
    for i in 1:nrow(df)
        add_edge!(G, df[i, :i], df[i, :j])
    end
    return G
end

races = ["Sakhir", "Jeddah", "Melbourne", "Shanghai", "Baku", "Miami", "Imola", "Monaco", "Barcelona", "Montreal", "Spielberg", "Silverstone", "Budapest", "Spa", "Zandvoort", "Monza", "Singapore", "Suzuka", "Lusail", "Austin", "Mexico City", "Sao Paulo", "Las Vegas","Yas Marina"]
races_circuitref = ["bahrain", "jeddah", "albert_park", "shanghai", "baku", "miami", "imola", "monaco", "catalunya", "villeneuve", "red_bull_ring", "silverstone", "hungaroring", "spa", "zandvoort", "monza", "marina_bay", "suzuka", "losail", "americas", "rodriguez", "interlagos", "las_vegas", "yas_marina"];

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

# create a distance matrix for each team home Base
dist_matrices = []
for team in eachrow(teams_df)
    df_temp = copy(df)
    team_name = team[:Team]
    team_home = team[:Home]
    team_lat = team[:lat]
    team_lng = team[:lng]
    # add a row to df for this team
    push!(df_temp, [team_name, team_home, team_lat, team_lng])
    # create a distance matrix for this team
    dist_mat = [dist(i, j, df_temp) for i in 1:size(df_temp, 1), j in 1:size(df_temp, 1)]
    push!(dist_matrices, dist_mat)
end

# add all the distance matrices together
dist_matrix = sum(dist_matrices)

# take the average of the distance matrix
distance_matrix = dist_matrix / size(teams_df, 1)

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

races = 25
home = 1
TH = 2
DH = 6
subloops = 14

# get first #races circuitRef from initial df
circuitRefs = df[1:races-1, :circuitRef]

# create vector with string "home" and add circuitRefs to it
circuits = ["home"; circuitRefs];

V = 1:races
V_0 = 2:races
all_S = Combinatorics.powerset(V_0, 2, length(V) - 2);

all_S = collect(all_S);

model = Model(Gurobi.Optimizer)
set_optimizer_attribute(model, "OutputFlag", 1)
set_optimizer_attribute(model, "Threads", 24)
#set_optimizer_attribute(model, "MIPGap", 0.005)
#set_optimizer_attribute(model, "TimeLimit", 600)

# variables
@variable(model, x[i in V, j in V], Bin)
#@variable(model, subloops >= 1)
@variable(model, double_header[i in V_0, j in V_0], Bin)
@variable(model, triple_header[i in V_0, j in V_0, q in V_0], Bin)

# constraints
# each circuit can only be visited once
@constraint(model, only_one_in[j in V_0], sum(x[i, j] for i in V) == 1)
@constraint(model, only_one_out[i in V_0], sum(x[i, j] for j in V) == 1) 

# we cannot go from a circuit to itself
@constraint(model, no_self_connect[i in V], x[i, i] == 0)

# we must start and end at home and the number of loops away from home is subloops
@constraint(model, K_petals_in, sum(x[i, 1] for i in V_0) == subloops)
@constraint(model, K_petals_out, sum(x[1, j] for j in V_0) == subloops)

# no subloops that do not connect to home
#p = Progress(length(all_S))
#Threads.@threads for S in all_S
for S in all_S
    if S != V_0
        @constraint(model, sum(x[i, j] for i in S, j in S) <= length(S) - 1)
    end
#    next!(p)
end

# define double_header as: 
# double_header[i, j] == 1 <=> x[home, i] == 1 and x[i, j] == 1 and x[j, home] == 1
@constraint(model, [i in V_0, j in setdiff(V_0, Set([i]))], double_header[i, j] <= x[home, i])
@constraint(model, [i in V_0, j in setdiff(V_0, Set([i]))], double_header[i, j] <= x[i, j])
@constraint(model, [i in V_0, j in setdiff(V_0, Set([i]))], double_header[i, j] <= x[j, home])
@constraint(model, [i in V_0, j in setdiff(V_0, Set([i]))], double_header[i, j] >= x[home, i] + x[i, j] + x[j, home] - 2)

# we want at most DH double-headers
@constraint(model, max_DH_doubleheaders, sum(double_header[i, j] for i in V_0, j in setdiff(V_0, Set([i]))) <= DH)

# define triple_header
# triple_header[i, j] == 1 <=> x[home, i] == 1 and x[i, j] == 1 and x[j, q] == 1 and x[q, home] == 1
@constraint(model, [i in V_0, j in setdiff(V_0, Set([i])), q in setdiff(V_0, Set([i, j]))], triple_header[i, j, q] <= x[home, i])
@constraint(model, [i in V_0, j in setdiff(V_0, Set([i])), q in setdiff(V_0, Set([i, j]))], triple_header[i, j, q] <= x[i, j])
@constraint(model, [i in V_0, j in setdiff(V_0, Set([i])), q in setdiff(V_0, Set([i, j]))], triple_header[i, j, q] <= x[j, q])
@constraint(model, [i in V_0, j in setdiff(V_0, Set([i])), q in setdiff(V_0, Set([i, j]))], triple_header[i, j, q] <= x[q, home])
@constraint(model, [i in V_0, j in setdiff(V_0, Set([i])), q in setdiff(V_0, Set([i, j]))], triple_header[i, j, q] >= x[home, i] + x[i, j] + x[j, q] + x[q, home] - 3)

# we want at most TH triple-headers
@constraint(model, max_TH_tripleheaders, sum(triple_header[i, j, q] for i in V_0, j in setdiff(V_0, Set([i])), q in setdiff(V_0, Set([i, j]))) <= TH)

# we want at most triple-headers, so no more than 3 consecutive races
@constraint(model, max_tripleheaders[i in V_0, j in setdiff(V_0, Set([i])), q in setdiff(V_0, Set([i, j])), r in setdiff(V_0, Set([i, j, q]))], x[home, i] + x[i, j] + x[j, q] + x[q, r] <= 3)

#objective
@objective(model, Min, sum(sum(x[i, j] * distance_matrix[i, j] for i in V) for j in V));

optimize!(model)

objective_value(model)

x_opt = value.(x);

df_full = convert_jump_container_to_df(x_opt, dim_names=[:i, :j], value_col=:x)
df_1 = df_full[df_full.x .== 1, :]
show(DataFrame([[names(df_1)]; collect.(eachrow(df_1))], [:column; Symbol.(axes(df_1, 1))]), allcols=true, allrows=true)