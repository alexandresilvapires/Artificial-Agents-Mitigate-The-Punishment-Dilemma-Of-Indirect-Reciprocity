include("./reputation.jl")
include("./strategy.jl")
using Test


tolerance = 1e-4
snIS = [1.0, 0.0, 1.0, 0.0] 
snSJ = [1.0, 0.0, 0.0, 1.0] 
snSS = [1.0, 0.0, 1.0, 1.0] 
snSH = [1.0, 0.0, 0.0, 0.0] 
snAG = [1.0, 1.0, 1.0, 1.0]
snAB = [0.0, 0.0, 0.0, 0.0]

@testset "Reputation Tests (no AAs)" begin

    begin # If everyone is AllC, in IS, everyone is good
        fallC, fallD, fDisc = 1.0, 0.0, 0.0
        sn = snIS
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 0
        result = avg_rep(fallC, fallD, fDisc, 0.0, 0, sns, pop, gossip)[1]
        @test isapprox(result, 1.0, atol=tolerance)
    end

    begin # If everyone is AllD, in IS, everyone is bad
        fallC, fallD, fDisc = 0.0, 1.0, 0.0
        sn = snIS
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 0

        result = avg_rep(fallC, fallD, fDisc, 0.0, 0, sns , pop, gossip)[2]
        @test isapprox(result, 0.0, atol=tolerance) 
    end

    begin # In a mixed population, in IS, every AllC is good and every AllD is bad
        fallC, fallD, fDisc = 0.2, 0.2, 0.6
        sn = snIS
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 0

        result = avg_rep(fallC, fallD, fDisc, 0.0, 0, sns, pop, gossip)[1]
        @test isapprox(result, 1.0, atol=tolerance)

        result = avg_rep(fallC, fallD, fDisc, 0.0, 0, sns, pop, gossip)[2]
        @test isapprox(result, 0.0, atol=tolerance)
    end

    begin # In a Disc population, in SJ with errors, reputations without gossip should be around 0.5
        fallC, fallD, fDisc = 0.0, 0.0, 1.0
        sn = add_errors_sn(snSJ, 0.02, 0.02)
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 0

        result = avg_rep(fallC, fallD, fDisc, 0.0, 0, sns, pop, gossip)[3]

        @test 0.4 < result < 0.6
    end

    begin # In a Disc population, in SJ with errors, reputations with some gossip should follow the values from the article
        fallC, fallD, fDisc = 0.0, 0.0, 1.0
        sn = add_errors_sn(snSJ, 0.02, 0.02)
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 20

        result = avg_rep(fallC, fallD, fDisc, 0.0, 0, sns, pop, gossip)[3]

        @test 0.65 < result < 0.75
    end

    begin # In a Disc population, in SJ with errors, agreement without gossip should be around 0.5
        fallC, fallD, fDisc = 0.0, 0.0, 1.0
        sn = add_errors_sn(snSJ, 0.02, 0.02)
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 0

        result = agreement_H_H(fallC, fallD, fDisc, 0.0, 0, sns, pop, gossip)

        @test 0.4 < result < 0.6
    end

    begin # In mixed population, agreement + disagreement = 1
        fallC, fallD, fDisc = 0.25, 0.35, 0.4
        sn = add_errors_sn(snSJ, 0.02, 0.02)
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 2

        result1 = agreement_H_H(fallC, fallD, fDisc, 0.0, 0, sns, pop, gossip)
        result2 = disagreement_H_H(fallC, fallD, fDisc, 0.0, 0, sns, pop, gossip)

        @test isapprox(result1 + result2, 1.0, atol=tolerance) 
    end

    begin # In a Disc population, in SJ with errors, if everyone gossips a lot, the agreement is very high
        fallC, fallD, fDisc = 0.0, 0.0, 1.0
        sn = add_errors_sn(snSJ, 0.01, 0.01)
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 5000000

        result = agreement_H_H(fallC, fallD, fDisc, 0.0, 0, sns, pop, gossip)
        @test result > 0.95
    end

    begin # If everyone is AllD, in IS, and the assessment and execution error is 0.5, half should be good
        fallC, fallD, fDisc = 0.0, 1.0, 0.0
        sn = add_errors_sn(snIS, 0.5, 0.5)
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 0

        result = avg_rep(fallC, fallD, fDisc, 0.0, 0, sns, pop, gossip)[2]
        @test isapprox(result, 0.5, atol=tolerance) 
    end

    begin # No matter the population, in SJ, and the assessment and execution error is 0.5, half should be good
        fallC, fallD, fDisc = 0.3, 0.2, 0.5
        sn = add_errors_sn(snSJ, 0.5, 0.5)
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 0

        result = avg_rep(fallC, fallD, fDisc, 0.0, 0, sns, pop, gossip)[1]
        @test isapprox(result, 0.5, atol=tolerance) 

        result = avg_rep(fallC, fallD, fDisc, 0.0, 0, sns, pop, gossip)[2]
        @test isapprox(result, 0.5, atol=tolerance) 

        result = avg_rep(fallC, fallD, fDisc, 0.0, 0, sns, pop, gossip)[3]
        @test isapprox(result, 0.5, atol=tolerance) 
    end

end

@testset "Payoff Tests (no AAs)" begin

    begin # If everyone is AllC, in IS, the average payoff of AllC should be b - c
        fallC, fallD, fDisc = 1.0, 0.0, 0.0
        sn = snIS
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 0
        execError = 0.0
        b = 5.0

        result = fit_allc(fallC, fallD, fDisc, 0.0, 0, execError, b, sns, pop, gossip)
        @test isapprox(result, b - 1, atol=tolerance)
    end

    begin # If everyone is AllD, in IS, the average payoff of AllD should be 0
        fallC, fallD, fDisc = 0.0, 1.0, 0.0
        sn = snIS
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 0
        execError = 0.0
        b = 5.0

        result = fit_alld(fallC, fallD, fDisc, 0.0, 0, execError, b, sns, pop, gossip)
        @test isapprox(result, 0.0, atol=tolerance)
    end

    begin # If everyone is Disc, in All Good, the average payoff is b - c
        fallC, fallD, fDisc = 0.0, 0.0, 1.0
        sn = snAG
        sns = [sn,sn,sn,sn,sn]
        pop = 50
        gossip = 0
        execError = 0.0
        b = 5.0

        result = fit_disc(fallC, fallD, fDisc, 0.0, 0, execError, b, sns, pop, gossip)
        @test isapprox(result, b - 1 , atol=tolerance)
    end

end

@testset "Evolutionary Tests (no AAs)" begin

    begin # In SJ, without gossip, the cooperation index should be very low
        sn = snSJ
        sns = [sn,sn,sn,sn,sn]
        pop = 30
        gossip = 0
        execError = 0.00
        b = 5.0

        result = coop_index(sns, 0.0, 0, pop, execError, b, 0.01, gossip)[1][1]
        @test result < 0.05
    end

    begin # In SJ, without gossip, the average state should be of full defection
        sn = snSJ
        sns = [sn,sn,sn,sn,sn]
        pop = 30
        gossip = 0
        execError = 0.00
        b = 5.0

        result = avg_state(sns, 0.0, 0, pop, execError, b, 0.01, gossip)
        @test result[2] > pop * 0.95
    end

    begin # In SJ, with gossip, the average state should no longer be full of defectors
        sn = snSJ
        sns = [sn,sn,sn,sn,sn]
        pop = 30
        gossip = 30
        execError = 0.00
        b = 5.0

        result = avg_state(sns, 0.0, 0, pop, execError, b, 0.01, gossip)
        @test result[2] < 0.6 * pop
    end

    begin # Without AAs, the cooperation index H_H, _H and All should be the same
        sn = snSJ
        sns = [sn,sn,sn,sn,sn]
        pop = 30
        gossip = 2
        execError = 0.00
        b = 5.0

        result = coop_index(sns, 0.0, 0, pop, execError, b, 0.01, gossip)[1]
        @test isapprox(result[1], result[4] , atol=tolerance) #coop_H_H and coop_H
        @test isapprox(result[4], result[5] , atol=tolerance) #coop_H and coop_All
    end

end

@testset "Evolutionary Tests (with AAs)" begin
    begin # In IS, if a mixed population interacts only with robots, and the robots always defect, the payoff of all populations is correct
        fallC, fallD, fDisc = 0.35, 0.3, 0.35
        sn = snIS
        sns = [sn,sn,sn,sn,sn]
        pop = 30
        gossip = 30
        execError = 0.00
        b = 5.0
        interactionsAA = 1.0
        stratAA = 1 # allD

        result = fit_allc(fallC, fallD, fDisc, interactionsAA, stratAA, execError, b, sns, pop, gossip)
        @test isapprox(result, -1.0, atol=tolerance) # payoff = -c 

        result = fit_alld(fallC, fallD, fDisc, interactionsAA, stratAA, execError, b, sns, pop, gossip)
        @test isapprox(result, 0.0, atol=tolerance) # payoff = 0

        result = fit_disc(fallC, fallD, fDisc, interactionsAA, stratAA, execError, b, sns, pop, gossip)
        @test isapprox(result, 0.0, atol=tolerance) # payoff = 0 
    end

end

@testset "Evolutionary Tests on reputations (with AAs)" begin
    begin # After evolution, if humans think all AAs are good (and judge humans with SJ, for example, for mixed reputations), and AAs think all humans are bad, the average reputations are all correct
        sns = [snSJ,snAG,snSJ,snAB,snAB]
        pop = 30
        gossip = 0
        execError = 0.00
        b = 5.0
        interactionsAA = 0.2
        stratAA = utils.STRAT_DISC # disc

        #rAllC_H, rAllD_H, rDisc_H, rAllC_AA, rAllD_AA, rDisc_AA, rAA_H
        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][7]
        @test isapprox(result, 1.0, atol=tolerance) # humans think AAs are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][4]
        @test isapprox(result, 0.0, atol=tolerance) # AAs think all humans are bad

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][5]
        @test isapprox(result, 0.0, atol=tolerance) # AAs think all humans are bad

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][6]
        @test isapprox(result, 0.0, atol=tolerance) # AAs think all humans are bad
    end

    begin # After evolution, if humans think all AAs are good (and judge humans with SJ, for example, for mixed reputations), and AAs think all humans are bad, the average reputations are all correct -> with errors and gossip effect remains
        sns = [snSJ,snAG,snSJ,snAB,snAB]
        pop = 30
        gossip = 2
        execError = 0.10
        b = 5.0
        interactionsAA = 0.2
        stratAA = utils.STRAT_DISC # disc

        #rAllC_H, rAllD_H, rDisc_H, rAllC_AA, rAllD_AA, rDisc_AA, rAA_H
        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][7]
        @test isapprox(result, 1.0, atol=tolerance) # humans think AAs are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][4]
        @test isapprox(result, 0.0, atol=tolerance) # AAs think all humans are bad

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][5]
        @test isapprox(result, 0.0, atol=tolerance) # AAs think all humans are bad

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][6]
        @test isapprox(result, 0.0, atol=tolerance) # AAs think all humans are bad
    end

    begin # After evolution, if humans think all AAs are bad (and judge humans with SJ, for example, for mixed reputations), and AAs think all humans are good, the average reputations are all correct
        sns = [snSJ,snAB,snSJ,snAG,snAG]
        pop = 30
        gossip = 0
        execError = 0.00
        b = 5.0
        interactionsAA = 0.2
        stratAA = utils.STRAT_DISC # disc

        #rAllC_H, rAllD_H, rDisc_H, rAllC_AA, rAllD_AA, rDisc_AA, rAA_H
        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][7]
        @test isapprox(result, 0.0, atol=tolerance) # humans think AAs are bad

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][4]
        @test isapprox(result, 1.0, atol=tolerance) # AAs think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][5]
        @test isapprox(result, 1.0, atol=tolerance) # AAs think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][6]
        @test isapprox(result, 1.0, atol=tolerance) # AAs think all humans are good
    end

    begin # After evolution, if humans think all AAs are bad (and judge humans with SJ, for example, for mixed reputations), and AAs think all humans are good, the average reputations are all correct -> with errors and gossip effect remains
        sns = [snSJ,snAB,snSJ,snAG,snAG]
        pop = 30
        gossip = 2
        execError = 0.10
        b = 5.0
        interactionsAA = 0.2
        stratAA = utils.STRAT_DISC # disc

        #rAllC_H, rAllD_H, rDisc_H, rAllC_AA, rAllD_AA, rDisc_AA, rAA_H
        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][7]
        @test isapprox(result, 0.0, atol=tolerance) # humans think AAs are bad

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][4]
        @test isapprox(result, 1.0, atol=tolerance) # AAs think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][5]
        @test isapprox(result, 1.0, atol=tolerance) # AAs think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][6]
        @test isapprox(result, 1.0, atol=tolerance) # AAs think all humans are good
    end

    begin # After evolution, if humans think all humans are good (and judge AAs with SS), and AAs judge humans with SS, the average reputations are all correct
        sns = [snAG,snSS,snAG,snSS,snSS]
        pop = 30
        gossip = 0
        execError = 0.00
        b = 5.0
        interactionsAA = 0.2
        stratAA = utils.STRAT_DISC # disc

        #rAllC_H, rAllD_H, rDisc_H, rAllC_AA, rAllD_AA, rDisc_AA, rAA_H
        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][1]
        @test isapprox(result, 1.0, atol=tolerance) # humans think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][2]
        @test isapprox(result, 1.0, atol=tolerance) # humans think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][3]
        @test isapprox(result, 1.0, atol=tolerance) # humans think all humans are good
    end

    begin # After evolution, if humans think all humans are good (and judge AAs with SS), and AAs judge humans with SS, the average reputations are all correct -> with errors
        sns = [snAG,snSS,snAG,snSS,snSS]
        pop = 30
        gossip = 2
        execError = 0.10
        b = 5.0
        interactionsAA = 0.2
        stratAA = utils.STRAT_DISC # disc

        #rAllC_H, rAllD_H, rDisc_H, rAllC_AA, rAllD_AA, rDisc_AA, rAA_H
        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][1]
        @test isapprox(result, 1.0, atol=tolerance) # humans think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][2]
        @test isapprox(result, 1.0, atol=tolerance) # humans think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][3]
        @test isapprox(result, 1.0, atol=tolerance) # humans think all humans are good
    end


    begin # After evolution, if AAs think humans are always bad, then their average reputation will be close to 0 -> with errors
        sns = [snSJ,snIS,snIS,snAB,snAB] # snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA
        pop = 100
        gossip = 2
        execError = 0.10
        b = 5.0
        interactionsAA = 0.3
        stratAA = utils.STRAT_DISC # disc

        #rAllC_H, rAllD_H, rDisc_H, rAllC_AA, rAllD_AA, rDisc_AA, rAA_H
        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][4]
        @test isapprox(result, 0.0, atol=tolerance) # AA think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][5]
        @test isapprox(result, 0.0, atol=tolerance) # AA think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][6]
        @test isapprox(result, 0.0, atol=tolerance) # AA think all humans are good
    end

    begin # After evolution, if AAs think humans are always good, then their average reputation will be close to 0 -> with errors
        sns = [snSJ,snIS,snIS,snAG,snAG] # snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA
        pop = 100
        gossip = 2
        execError = 0.23
        b = 5.0
        interactionsAA = 0.5
        stratAA = utils.STRAT_DISC # disc

        #rAllC_H, rAllD_H, rDisc_H, rAllC_AA, rAllD_AA, rDisc_AA, rAA_H
        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][4]
        @test isapprox(result, 1.0, atol=tolerance) # AA think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][5]
        @test isapprox(result, 1.0, atol=tolerance) # AA think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][6]
        @test isapprox(result, 1.0, atol=tolerance) # AA think all humans are good
    end

    begin # After evolution, if AllC AAs think humans are always bad, then their average reputation will be close to 0 -> with errors
        sns = [snSJ,snIS,snIS,snAB,snAB] # snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA
        pop = 100
        gossip = 2
        execError = 0.10
        b = 5.0
        interactionsAA = 0.3
        stratAA = utils.STRAT_ALLC # allc

        #rAllC_H, rAllD_H, rDisc_H, rAllC_AA, rAllD_AA, rDisc_AA, rAA_H
        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][4]
        @test isapprox(result, 0.0, atol=tolerance) # AA think all humans are bad

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][5]
        @test isapprox(result, 0.0, atol=tolerance) # AA think all humans are bad

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][6]
        @test isapprox(result, 0.0, atol=tolerance) # AA think all humans are bad
    end

    begin # After evolution, if AllD AAs think humans are always good, then their average reputation will be close to 0 -> with errors
        sns = [snSJ,snIS,snIS,snAG,snAG] # snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA
        pop = 100
        gossip = 2
        execError = 0.23
        b = 5.0
        interactionsAA = 0.5
        stratAA = utils.STRAT_ALLD # alld

        #rAllC_H, rAllD_H, rDisc_H, rAllC_AA, rAllD_AA, rDisc_AA, rAA_H
        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][4]
        @test isapprox(result, 1.0, atol=tolerance) # AA think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][5]
        @test isapprox(result, 1.0, atol=tolerance) # AA think all humans are good

        result = avg_reputations(sns, interactionsAA, stratAA, pop, execError, b, 0.01, gossip)[1][6]
        @test isapprox(result, 1.0, atol=tolerance) # AA think all humans are good
    end
end