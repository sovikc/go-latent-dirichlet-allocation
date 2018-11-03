package main

//#cgo LDFLAGS: -L . ./randomkit.so
//#cgo LDFLAGS: -L . ./distributions.so
//#cgo LDFLAGS: -L . ./psi.so
//#cgo LDFLAGS: -L . ./gamma.so
//#include "randomkit.h"
//#include "distributions.h"
//#include "mconf.h"
import "C"
import (
	"fmt"
	"sort"
	"errors"
	"math"
	"reflect"
)
//--------------------------------------------------------------------------------------------------
// The Dictionary
//--------------------------------------------------------------------------------------------------
type Document struct {
	Words []string
}

type Dictionary struct {
	Documents [][]string
	//token -> tokenId
	Token2id map[string]int
	// reverse mapping for token2id; only formed on request, to save memory
	Id2token map[int]string
	// document frequencies: tokenId -> in how many documents this token appeared
	DocumentFrequency map[int]int
	//document frequencies: tokenId -> in how many documents this token appeared
	DFS map[int]int
	// number of documents processed
	Num_Docs int
	// total number of corpus positions
	Num_Pos int
	// total number of non-zeroes in the BOW matrix
	Mum_Nnz int
	// AllowUpdate
	AllowUpdate bool
	// ReturnMissing
	ReturnMissing bool
}

// NewDictionary - returns a new Dictionary
func NewDictionary(docs [][]string) (ptrDict *Dictionary) {
	token2id := make(map[string]int, 0)
	id2token := make(map[int]string, 0)
	dfs := make(map[int]int, 0)

	dictionary := &Dictionary{Documents: docs, Token2id: token2id, DFS: dfs, Id2token: id2token}

	if docs != nil {
		dictionary.AddDocuments(docs)
	}
	return dictionary

}

func (dict *Dictionary) AddDocuments(documents [][]string) {

	for i := 0; i < len(documents); i++ {
		// ignore the result, here we only care about updating token ids
		dict.doc2bow(documents[i], true, false)
		//fmt.Print(documents[i])
	}
}

// doc2bow converts a dictionary into bag of words
func (dict *Dictionary) doc2bow(doc []string, allowUpdate bool, returnMissing bool) (map[int]int, map[string]int) {

	//Construct (word, frequency) mapping.
	//counter = defaultdict(int)
	counter := make(map[string]int, 0)
	missing := make(map[string]int, 0)

	result := make(map[int]int, 0)
	sorted := make(map[int]int, 0)



	for _, word := range doc {
		found, times := ifTokenExists(word, doc)
		if found == true {
			counter[word] = times
		}
	}

	/*fmt.Println("doc")
	fmt.Println(doc)
	fmt.Println("counter")
	fmt.Println(counter)*/

	if allowUpdate || returnMissing {
		//missing = dict((w, freq) for w, freq in iteritems(counter) if w not in token2id)
		for w, freq := range counter {
			if _, ok := dict.Token2id[w]; !ok {
				missing[w] = freq
			}
		}

		/*fmt.Println("missing")
		fmt.Println(missing)*/

		if allowUpdate {
			for w, _ := range missing {
				//fmt.Println("before", w, len(dict.Token2id))
				dict.Token2id[w] = len(dict.Token2id)
				//fmt.Println("after", w, len(dict.Token2id))
			}
		}

	}

	for w, freq := range counter {
		if _, ok := dict.Token2id[w]; ok {
			result[dict.Token2id[w]] = freq
		}
	}


	for key, value := range dict.Token2id {
		dict.Id2token[value] = key
	}


	if allowUpdate {
		dict.Num_Docs = dict.Num_Docs + 1
		dict.Num_Pos = dict.Num_Pos + 1
		dict.Mum_Nnz = len(result)
		// increase document count for each unique token that appeared in the document
		for tokenId, _ := range result {
			docFreq := 0
			if val, ok := dict.DFS[tokenId]; ok {
				docFreq = val
			}
			dict.DFS[tokenId] = docFreq + 1
		}

		// return tokenids, in ascending id order
		keys := make([]int, len(result))
		for key, _ := range result {
			keys = append(keys, key)
		}
		sort.Ints(keys)

		for _, key := range keys {
			sorted[key] = result[key]
		}
	}
	return sorted, missing
}

func ifTokenExists(w string, doc []string) (bool, int) {
	found := false
	times := 0
	for _, item := range doc {
		if w == item {
			found = true
			times = times + 1
		}
	}
	return found, times
}

func (dict *Dictionary) keys() []int{
	keys := make([]int, 0)
	for _, v := range dict.Token2id {
		// ignore the result, here we only care about updating token ids
		keys = append(keys, v)
	}
	return keys
}
//--------------------------------------------------------------------------------------------------
// The LDA State
//--------------------------------------------------------------------------------------------------

type LDAState struct {
	ETA []float64
	NumDocs int
	NumTopics int
	NumTerms int
	SStats [][]float64
	SStatesShape []int
}

func NewLDAState(ETA []float64, NumTopics int, NumTerms int) (*LDAState) {
	return &LDAState{ETA: ETA, NumTopics: NumTopics, NumTerms: NumTerms}
}

func (st *LDAState) GetLambda() [][]float64 {
	fmt.Println("st.ETA ", len(st.ETA))
	fmt.Println("len(st.SStates)", len(st.SStats))
	// merge them two matrices by summing
	//lambda := st.ETA + st.SStats

	lambda := make([][]float64, len(st.SStats))

	for i := 0; i < len(lambda); i++ {
		innerLen := len(st.SStats[i])
		lambda[i] = make([]float64, innerLen)
	}

	for i := 0; i < len(lambda); i++ {
		innerLen := len(lambda)
		lambda[i] = make([]float64, innerLen)
		for j := 0; j < innerLen; j++ {
			temp := st.ETA[i] + st.SStats[i][j]
			lambda[i] = append(lambda[i], temp)
		}
	}
	return lambda
}

func (st *LDAState) Blend(rho float64, other *LDAState, targetSize int) {

	scale := 1.0
	if targetSize == 0 {
		targetSize = st.NumDocs
	}
	// stretch the current model's expected n*phi counts to target size
	if st.NumDocs == 0 || targetSize == st.NumDocs {
		scale = 1.0
	} else {
		scale = float64(1.0 * targetSize / st.NumDocs)
	}

	//st.SStats *= (1.0 - rho) * scale
	scaleChange := (1.0 - rho) * scale
	MultMS(st.SStats, scaleChange)

	// stretch the incoming n*phi counts to target size
	if other.NumDocs == 0 || targetSize == other.NumDocs {
		scale = 1.0
	} else {
		fmt.Print("merging changes from %i documents into a model of %i documents", other.NumDocs, targetSize)
		scale = float64(1.0 * targetSize / other.NumDocs)
	}
}

func (st *LDAState) GetElogbeta() [][]float64 {
	return DirichletExpectation2D(st.GetLambda())
}


//--------------------------------------------------------------------------------------------------
// The LDA Params
//--------------------------------------------------------------------------------------------------

type LDAParams struct {
	Corpus []map[int]int
	NumTopics int
	Id2Word *Dictionary //map[int]string
	Distributed bool
	ChunkSize int
	Passes int
	UpdateEvery int
	AlphaType string
	ETAType string
	Decay float64
	Offset float64
	EvalEvery int
	Iterations int
	GammaThreshold float64
	MinimumProbability float64
	RandomState int
	NsConf	map[string]string
	MinimumPhiValue float64
	PerWordTopics bool
}

func NewLDAParams() (*LDAParams) {
	corpus := make([]map[int]int, 0)
	return &LDAParams{Corpus:corpus}
}
//--------------------------------------------------------------------------------------------------
// The LDA Model
//--------------------------------------------------------------------------------------------------
type LDAModel struct {

	Corpus []map[int]int
	NumTopics int

	Id2Word      *Dictionary
	Distributed bool
	ChunkSize int
	Passes       int
	UpdateEvery int

	AlphaType      string
	ETAType        string
	Decay      float64
	OffSet     float64
	EvalEvery int
	Iterations int

	GammaThreshold     float64
	MinimumProbability float64
	RandomState        *C.struct_rk_state_
	NsConf             map[string]string
	MinimumPhiValue   float64
	PerWordTopics     bool

	NumTerms int
	NumUpdates int
	OptimizedAlpha bool
	OptimizedETA bool

	Alpha []float64
	ETA []float64

	NumWorkers int

	State *LDAState
	ExpElogbeta [][]float64
}

func NewLDAModel(ldaParams *LDAParams) (lda *LDAModel) {
	return &LDAModel{}
}

func (lda *LDAModel) BuildModel(ldaParams *LDAParams) error {

	if ldaParams.Corpus == nil && ldaParams.Id2Word == nil {
		err := errors.New("at least one of corpus/id2word must be specified, to establish input space dimensionality")
		return err
	}

	lda.Corpus = ldaParams.Corpus

	lda.Id2Word = ldaParams.Id2Word

	if len(lda.Id2Word.keys()) > 0 {
		maxKey := 0
		for key, _ := range lda.Id2Word.keys() {
			if key > maxKey {
				maxKey = key
			}
		}
		lda.NumTerms = 1 + maxKey
	} else {
		lda.NumTerms = 0
	}

	if lda.NumTerms == 0 {
		err := errors.New("cannot compute LDA over an empty collection (no terms)")
		return err
	}

	lda.NumTopics = ldaParams.NumTopics
	lda.Distributed = false
	lda.ChunkSize = 2000
	//lda.ChunkSize = 10
	lda.Passes = 1
	lda.UpdateEvery = 1
	lda.AlphaType = "symmetric"
	lda.ETAType = ""
	lda.Decay = 0.5
	lda.OffSet = 1.0
	lda.EvalEvery = 10
	lda.Iterations = 50
	lda.GammaThreshold = 0.001
	lda.MinimumProbability = 0.01
	lda.RandomState = &C.struct_rk_state_{}
	lda.NsConf = nil
	lda.MinimumPhiValue = 0.01
	lda.PerWordTopics = false

	return nil
}

func (lda *LDAModel) PopulateAlphaAndETA() error {

	alpha, optimizedAlpha, err := lda.initDirPrior(lda.AlphaType, "alpha")
	if err != nil {
		err := errors.New("something wrong happened in iniDirPrior while processing alpha")
		return err
	}
	lda.Alpha = alpha
	lda.OptimizedAlpha = optimizedAlpha

	fmt.Println("alpha ============================", len(lda.Alpha), lda.NumTopics)

	if  len(lda.Alpha) != lda.NumTopics {
		return errors.New("Invalid alpha shape.")
	}

	if lda.ETAType == "asymmetric" {
		err1 := errors.New("something wrong happened in iniDirPrior while processing eta")
		return err1
	}

	eta, optimizedETA, err := lda.initDirPrior(lda.ETAType, "eta")

	lda.ETA = eta
	lda.OptimizedETA = optimizedETA

	fmt.Println("eta ============================")

	//lda.RandomState = &C.struct_rk_state_{}
	rk_err := C.rk_randomseed(lda.RandomState)
	//err := C.rk_seed(1, st)
	fmt.Println("rk_err", rk_err)


	lda.State = NewLDAState(lda.ETA, lda.NumTopics, lda.NumTerms)
	lda.State.SStats = Gamma2D(lda.RandomState, 100., 1. / 100., []int{lda.NumTopics, lda.NumTerms})
	lda.ExpElogbeta = Exp(DirichletExpectation2D(lda.State.SStats))

	fmt.Println("lda.Corpus", lda.Corpus)
	if lda.Corpus != nil {
		lda.update(lda.Corpus, false)
	}

	return nil
}

func (lda *LDAModel) UpdateAlpha(gammat [][]float64, rho float64) []float64 {
	logphat := make([]float64, 0)
	N := float64(len(gammat))
	temp := make([][]float64, 0)
	sumResults := make([]float64, 0)
	for i := 0; i < len(gammat); i++ {
		temp = append(temp, DirichletExpectation(gammat[i]))
	}
	for i := 0; i <= len(temp); i++ {
		for j := 0; j <= len(temp); j++ {
			sumResults[j] = sumResults[j] + temp[i][j]
		}
	}
	for i := 0; i <= len(temp); i++ {
		for j := 0; j <= len(temp); j++ {
			sumResults[j] = sumResults[j] + temp[i][j]
		}
	}
	for i := 0; i <= len(temp); i++ {
		logphat = append(logphat, sumResults[i] / N)
	}
	lda.Alpha = UpdateDirPrior(lda.Alpha, N, logphat, rho)
	return lda.Alpha
}

func (lda *LDAModel) initDirPrior(prior string, name string) ([]float64, bool, error) {

	var temp_init_prior = make([]float64, 0)
	var init_prior = make([]float64, 0)
	var is_auto bool

	if prior == "" {
		prior = "symmetric"
	}
	prior_shape := 0
	if name == "alpha" {
		prior_shape = lda.NumTopics
		//init_prior = [lda.NumTopics]int{}
	} else if name == "eta" {
		prior_shape = lda.NumTerms
	} else {
		err := errors.New("'name' must be 'alpha' or 'eta'")
		return init_prior, is_auto, err
	}


	if prior == "symmetric" {
		for i := 0; i < prior_shape; i++ {
			fmt.Println("using symmetric at ", name)
			result := float64(1.0 / lda.NumTopics)
			init_prior = append(init_prior, result)
		}
	} else if prior == "asymmetric" {

		var sum float64
		for i := 0; i < prior_shape; i++ {
			result := float64(1.0 / (float64(i) + math.Sqrt(float64(prior_shape))))
			sum = sum + result
			temp_init_prior = append(temp_init_prior, result)
		}
		for _, v := range temp_init_prior {
			init_prior = append(init_prior, v / sum)
		}
	} else if prior == "auto" {
		/*
		init_prior = np.asarray([1.0 / self.num_topics for i in xrange(prior_shape)])
		if name == 'alpha':
			logger.info("using autotuned %s, starting with %s", name, list(init_prior))
		 */
		is_auto = true
		for i := 0; i < prior_shape; i++ {
			fmt.Println("using auto at ", name)
			result := float64(1.0 / prior_shape)
			init_prior = append(init_prior, result)
		}
		if name == "alpha" {
			fmt.Println("using autotuned starting with", name)
		}
	} else {
		err := errors.New("Unable to determine proper prior value")
		return init_prior, is_auto, err
	}

	return init_prior, is_auto, nil
}

func (lda *LDAModel) update(corpus []map[int]int, chunksAsNumpy bool) error{
	//decay := lda.Decay
	//offset := lda.OffSet
	passes := lda.Passes
	updateEvery := lda.UpdateEvery
	evalEvery := lda.EvalEvery
	//iterations := lda.Iterations
	//gammaThreshold := lda.GammaThreshold
	chunkSize := lda.ChunkSize

	updatAfter := 0
	//evalAfter := 0

	var dirty bool = false
	fmt.Print(dirty)
	updateType := ""
	if corpus == nil {
		return errors.New("Invalid Corpus to update.")
	}
	lenCorpus := len(corpus)

	if(chunkSize == 0) {
		chunkSize = min(lenCorpus, lda.ChunkSize)
	}
	lda.State.NumDocs = lda.State.NumDocs + lenCorpus
	if lda.UpdateEvery == 1 {
		updateType = "online"
		if lda.Passes == 1 {
			updateType += " (single-pass)"
		}
		updatAfter = min(lenCorpus, updateEvery * lda.NumWorkers * chunkSize)
	}
	//evalAfter := min(lenCorpus, (evalEvery || 0) * lda.NumWorkers * chunkSize)

	lda.NumWorkers = 1
	updatesPerPass := max(1, lenCorpus / updatAfter)

	if updatesPerPass * lda.Passes < 10 {
		fmt.Println("too few updates, training might not converge. Consider increasing the number of passes or iterations to improve accuracy")
	}

	for i := 0; i < passes; i++ {
		other := NewLDAState(lda.ETA, lda.NumTopics, lda.NumTerms)
		dirty = false
		/*
			for chunk_no, chunk in enumerate(utils.grouper(corpus, chunksize, as_numpy=chunks_as_numpy))
		*/
		var numberOfChunks = 0
		if lenCorpus % chunkSize > 0 {
			numberOfChunks = lenCorpus / chunkSize + 1
		} else {
			numberOfChunks = lenCorpus / chunkSize
		}

		chunks := make([][]map[int]int, 0)
		chunk := make([]map[int]int, 0)
		next := 0
		realLen := 0
		fmt.Println("corpus", len(corpus))
		fmt.Println("numberOfChunks", numberOfChunks)
		for i := 0; i < numberOfChunks; i++ {
			chunk, next = grouper(corpus, chunkSize, next)
			fmt.Println("chunk", len(chunk))
			chunks = append(chunks, chunk)
		}
		fmt.Println("chunks", chunks)

		for chunkNo, data :=  range chunks {
			// keep track of how many documents we've processed so far
			realLen += len(data)

			if evalEvery > 0 && ((realLen == lenCorpus) || ((chunkNo + 1) % (evalEvery * lda.NumWorkers) == 0)) {
				lda.logPerplexity(data, lenCorpus)
			}

			gammat := lda.DoEStep(chunk, other)
			dirty = true

			// rho is the "speed" of updating;
			_rho := func (offset float64, pass int, numUpdates int, chunkSize int, decay float64) float64 {
				return math.Pow(offset + float64(pass) + float64(numUpdates / chunkSize), -float64(decay))
			}(lda.OffSet , i, lda.NumUpdates, chunkSize, lda.Decay)

			if lda.OptimizedAlpha {
				lda.UpdateAlpha(gammat, func (offset float64, pass int, numUpdates int, chunkSize int, decay float64) float64 {
					return math.Pow(offset + float64(pass) + float64(numUpdates / chunkSize), -float64(decay))
				}(lda.OffSet , i, lda.NumUpdates, chunkSize, lda.Decay))
			}
			dirty = true
			chunk = nil

			_rho = func (offset float64, pass int, numUpdates int, chunkSize int, decay float64) float64 {
				return math.Pow(offset + float64(pass) + float64(numUpdates / chunkSize), -float64(decay))
			}(lda.OffSet , i, lda.NumUpdates, chunkSize, lda.Decay)

			//perform an M step. determine when based on update_every, don't do this after every chunk
			if ((updateEvery == 1) && ((chunkNo + 1) % (updateEvery * lda.NumWorkers)) == 0 ) {
				lda.DoMStep(_rho, other, i > 0)
				other = nil
			}

			shape := [2]int{}
			shape[0] = len(lda.State.SStats)
			for i = 0; i < len(lda.State.SStats); i++ {
				shape[1] = len(lda.State.SStats[i])
			}

			other = NewLDAState(lda.ETA, shape[0], shape[1])
			dirty = false
		}
		if realLen != lenCorpus {
			fmt.Println("input corpus size changed during training (don't use generators as input)")
		}
		if dirty {
			_rho := func (offset float64, pass int, numUpdates int, chunkSize int, decay float64) float64 {
				return math.Pow(offset + float64(pass) + float64(numUpdates / chunkSize), -float64(decay))
			}(lda.OffSet , i, lda.NumUpdates, chunkSize, lda.Decay)
			lda.DoMStep(_rho, other, i > 0)
			other = nil
			dirty = false
		}
	}
	return nil
}

func (lda *LDAModel) DoMStep(rho float64, other *LDAState, extraPass bool) {

	fmt.Print("updating topics")

	lda.State.Blend(rho, other, 0)
	//diff -= model.state.get_Elogbeta()
	lda.SyncState()

	// print out some debug info at the end of each EM iteration
	lda.PrintTopics(5, 10)
	//logger.info("topic diff=%f, rho=%f", np.mean(np.abs(diff)), rho)

	if lda.OptimizedETA {
		lda.UpdateETA(lda.State.GetLambda(), rho)
	}

	if extraPass {
		// only update if this isn't an additional pass
		lda.NumUpdates += other.NumDocs
	}

}

func (lda *LDAModel) UpdateETA (lambdat [][]float64, rho float64) []float64 {

	N := float64(len(lambdat))
	temp := make([][]float64, 0)
	sumResults := make([]float64, 0)
	for i := 0; i < len(lambdat); i++ {
		temp = append(temp, DirichletExpectation(lambdat[i]))
	}
	for i := 0; i <= len(temp); i++ {
		for j := 0; j <= len(temp); j++ {
			sumResults[j] = sumResults[j] + temp[i][j]
		}
	}
	logphat := Reshape(DivideVByScalar(sumResults, N), lda.NumTerms)

	lda.ETA = UpdateDirPrior(lda.ETA, N, logphat, rho)

	return lda.ETA
}

func UpdateDirPrior(prior []float64, N float64, logphat []float64, rho float64) []float64 {
	/*
	"""
    Updates a given prior using Newton's method, described in
    **Huang: Maximum Likelihood Estimation of Dirichlet Distribution Parameters.**
    http://jonathan-huang.org/research/dirichlet/dirichlet.pdf
    """
    dprior = np.copy(prior)
    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)

    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)

    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))

    dprior = -(gradf - b) / q

    if all(rho * dprior + prior > 0):
        prior += rho * dprior
    else:
        logger.warning("updated prior not positive")

    return prior
	 */

	dPrior := CopyVector(prior)
	gradf := SubtractSV(N * PsiScalar(Sum(prior)), AddV(PsiVector(prior), logphat))

	// c = N * polygamma(1, np.sum(prior))
	// q = -N * polygamma(1, prior)

	//b := Sum(gradf / q) / (1 / c + np.sum(1 / q))
}

func (lda *LDAModel) PrintTopics(numTopics int, numWords int) bool {

	return lda.ShowTopics(numTopics, numWords, true, true)
}

func (lda *LDAModel) ShowTopics(numTopics int, numWords int, log bool, formatter bool) bool {

	return true
}

func (lda *LDAModel) SyncState() {
	lda.ExpElogbeta = Exp(lda.State.GetElogbeta())
}


func (lda *LDAModel) DoEStep(chunk []map[int]int, state *LDAState) [][]float64 {


	gamma, sstats := lda.Inference(chunk, true)
	state.SStats = AddMM(state.SStats, sstats)
	state.NumDocs = state.NumDocs + len(gamma)  // avoids calling len(chunk) on a generator
	return gamma
}

func (lda *LDAModel) logPerplexity(chunk []map[int]int, totalDocs int) float64 {

	corpusWords := 0
	for _, doc := range chunk {
		for _, cnt := range doc {
			corpusWords = corpusWords + cnt
		}
	}

	fmt.Println("corpusWords", corpusWords)
	subSampleRatio := 1.0 * totalDocs / len(chunk)
	score := lda.Bound(chunk, float64(subSampleRatio))


	powerBound := score / float64(subSampleRatio * corpusWords)
	return powerBound
}

func (lda *LDAModel) Bound(corpus []map[int]int, subsampleRatio float64) float64 {
	score := 0.0
	subsampleRatio = 1.0
	_lambda := lda.State.GetLambda()
	Elogbeta := DirichletExpectation2D(_lambda)
	gammad :=  make([][]float64, 0)

	for d, doc := range corpus {
		if d % lda.ChunkSize == 0 {
			fmt.Print("bound: at document #%i", d)
		}
		if gamma == nil {
			chunk:= make([]map[int]int, 1)
			chunk = append(chunk, doc)
			gammad, _ = lda.Inference(chunk, false)
		}
		Elogthetad := DirichletExpectation2D(gammad)

		//score += np.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)

		tempScorePerDoc := make([]float64, len(doc))
		for id, cnt := range doc {
			tempScorePerDoc = append(tempScorePerDoc, float64(cnt) * LogSumExp(AddMV(Elogthetad, getMatrixColumnsById(Elogbeta, id))))
		}
		score := Sum(tempScorePerDoc)


		// E[log p(theta | alpha) - log q(theta | gamma)]; assumes alpha is a vector
		score = score + SumM(Multiply(SubtractVM(lda.Alpha, gammad), Elogthetad))



		score = score + SumM(SubtractMV(GammaLn2D(gammad), GammaLn(lda.Alpha)))
		score += GammaLnScalar((Sum(lda.Alpha)- GammaLnScalar(SumM(gammad))))

	}
	sumETA := Sum(lda.ETA)
	score = score + subsampleRatio


	score = score + SumM(Multiply(SubtractVM(lda.ETA, _lambda), Elogbeta))
	score = score + SumM(SubtractMV(GammaLn2D(_lambda), GammaLn(lda.ETA)))

	sumETA = Sum(lda.ETA)

	score = score + Sum(SubtractSV(GammaLnScalar(sumETA), GammaLn(SumAxis1(_lambda))))
	return score
}

func (lda *LDAModel) Inference(chunk []map[int]int, collectSStats bool) ([][]float64, [][]float64) {
	lenChunk := len(chunk)
	SStats := make([][]float64, 0)
	converged := 0
	// 	Initialize the variational distribution q(theta|gamma) for the chunk
	size := []int{lenChunk, lda.NumTopics}
	gamma := Gamma2D(lda.RandomState, 100., 1. / 100., size)
	Elogtheta := DirichletExpectation2D(gamma)
	expElogtheta := Exp(Elogtheta)

	if collectSStats {
		dim := make([]int, 2)
		dim[0] = len(lda.ExpElogbeta)
		dim[1] = func() int {
			row := lda.ExpElogbeta[0]
			return len(row)
		}()
		SStats = zerosLike(dim)
	} else {
		SStats = nil
		converged = 0
	}

	ids := make([]int, 0)
	cts := make([]int, 0)


	for docId, doc := range chunk {
		if len(doc) > 0 {
			// make sure the term IDs are ints, otherwise np will get upset
			for id, cnt := range doc {
				ids = append(ids, id)
				cts = append(cts, cnt)
			}
			/*for _, cnt := range doc {
				cts = append(cts, cnt)
			}*/
		}
		// gammad = gamma[d, :]
		gammad, err := getMatrixRows(gamma, docId)
		if err != nil {
			fmt.Println(err.Error())
		}
		// Elogthetad = Elogtheta[d, :]
		Elogthetad, err := getMatrixRows(Elogtheta, docId)
		if err != nil {
			fmt.Println(err.Error())
		}

		// expElogthetad = expElogtheta[d, :]
		expElogthetad, err := getMatrixRows(expElogtheta, docId)
		if err != nil {
			fmt.Println(err.Error())
		}

		// expElogbetad = self.expElogbeta[:, ids]
		expElogbetad, err := getMatrixColumnsByIds(expElogtheta, ids)
		if err != nil {
			fmt.Println(err.Error())
		}

		phinorm := DotVM(expElogthetad, expElogbetad) //+ 1e-100

		expElogbetadT := transposeM(expElogbetad)
		expElogthetadT := transposeV(expElogthetad)


		for i:=0; i < lda.Iterations; i++ {
			lastgamma := gammad

			gammad = AddV(lda.Alpha, MultiplyV(expElogthetad, DotVM(DivideV(cts, phinorm), expElogbetadT)))

			Elogthetad = DirichletExpectation(gammad)
			expElogthetad = Exp1D(Elogthetad)
			phinorm = AddVerySmallNumber(DotVM(expElogthetad, expElogbetad))
			// If gamma hasn't changed much, we're done.
			meanChange := MeanV(AbsV(SubtractV(gammad, lastgamma)))
			if (meanChange < lda.GammaThreshold) {
				converged += 1
				break
			}
		}
		// gamma[d, :] = gammad
		SetMatrixFromVector(docId, gamma, gammad)

		if collectSStats {
			// Contribution of document d to the expected sufficient
			// statistics for the M step.
			values := Outer(expElogthetadT, DivideV(cts, phinorm))
			SetSStasRowsForDocIds(ids, SStats, values)
		}
	}
	if collectSStats {

		SStats = Multiply(SStats, lda.ExpElogbeta)
	}

	return gamma, SStats
}

func grouper(corpus []map[int]int, chunkSize int, next int) ([]map[int]int, int) {

	var length = next + chunkSize
	if length > len(corpus) {
		length = len(corpus)
	}
	wrappedChunk := corpus[next:length]
	fmt.Println("wrappedChunk", next, length, wrappedChunk)
	next = next + chunkSize
	return wrappedChunk, next
}


func (lda *LDAModel) updateAsChunks(Corpus map[int]int, chunkSize int,
Decay float64, offSet float64, passes int, updateEvery int,
iterations int, GammaThreshold float64, chunksAsNumpy bool) {
}

//----------------- ---------------------------------------------------------------------------------
// Utils
//--------------------------------------------------------------------------------------------------

func dictFromCorpus(dictionary Dictionary) *Dictionary {
	return nil
}

func Add(A [][]float64, B [][]float64) [][]float64 {
	result := make([][]float64, len(A))

	for i := 0; i < len(A); i++ {
		innerLen := len(A[i])
		result[i] = make([]float64, innerLen)
		for j := 0; j < innerLen; j++ {
			temp := A[i][j] + B[i][j]
			result[i] = append(result[i], temp)
		}
	}
	return result
}

func Subtract(A [][]float64, B [][]float64) [][]float64 {
	result := make([][]float64, len(A))

	for i := 0; i < len(A); i++ {
		innerLen := len(A[i])
		result[i] = make([]float64, innerLen)
		for j := 0; j < innerLen; j++ {
			temp := A[i][j] - B[i][j]
			result[i] = append(result[i], temp)
		}
	}
	return result
}

func SubtractVM(A []float64, B [][]float64) [][]float64 {
	result := make([][]float64, len(A))

	for i := 0; i < len(A); i++ {
		innerLen := len(B[i])
		result[i] = make([]float64, innerLen)
		for j := 0; j < innerLen; j++ {
			temp := A[i] - B[i][j]
			result[i] = append(result[i], temp)
		}
	}
	return result
}

func SubtractV(A []float64, B []float64) []float64 {
	result := make([]float64, len(A))

	for i := 0; i < len(A); i++ {
		temp := A[i] - B[i]
		result = append(result, temp)
	}
	return result
}

func SubtractSV(A float64, B []float64) []float64 {
	result := make([]float64, len(B))

	for i := 0; i < len(B); i++ {
		temp := A - B[i]
		result = append(result, temp)
	}
	return result
}

func SubtractMV(A [][]float64, B []float64) [][]float64 {
	result := make([][]float64, len(A))

	for i := 0; i < len(B); i++ {
		val := B[i]
		row := A[i]
		for j := 0; j < len(row); j++ {
			result[i][j] = val - A[i][j]
		}
	}
	return result
}

func AbsV(A []float64) []float64 {
	result := make([]float64, len(A))

	for i := 0; i < len(A); i++ {
		temp := math.Abs(A[i])
		result = append(result, temp)
	}
	return result
}

func Mean(A [][]float64) float64 {
	mean := 0.0
	sum := 0.0
	num := 0.0

	for i := 0; i < len(A); i++ {
		innerLen := len(A[i])
		for j := 0; j < innerLen; j++ {
			sum = sum + A[i][j]
			num = num + 1
		}
	}
	mean = sum / num
	return mean
}

func MeanV(A []float64) float64 {
	mean := 0.0
	sum := 0.0
	num := 0.0

	for i := 0; i < len(A); i++ {
		sum = sum + A[i]
		num = num + 1
	}
	mean = sum / num
	return mean
}


func gamma(shape float64, scale float64, numTopics int, numTerms int) []float64 {
	sstats := make([]float64, 0)
	fmt.Println("numTopics * numTerms", numTopics * numTerms)
	st := &C.struct_rk_state_{}
	err := C.rk_randomseed(st)
	fmt.Println("err", err)
	for i := 0; i < numTopics * numTerms; i++ {
		fmt.Println("after struct_rk_state_")
		result := C.rk_gamma(st, 100., 1./100.)
		fmt.Println(float64(result), reflect.TypeOf(result))
		sstats = append(sstats, float64(result))
	}
	return sstats
}

func Gamma2D(st *C.struct_rk_state_, shape float64, scale float64, size[] int) [][]float64 {
	sstats := make([][]float64, 0)
	twoD := make([][]float64, size[0])
	newSize := size[0] * size[1]

	fmt.Println("numTopics * numTerms", newSize)


	for i := 0; i < size[0]; i++ {
		innerLen := size[1]
		twoD[i] = make([]float64, innerLen)
		for j := 0; j < innerLen; j++ {
			result := C.rk_gamma(st, 100., 1./100.)
			fmt.Println(float64(result), reflect.TypeOf(result))
			twoD[i][j] = float64(result)
		}
	}
	sstats = twoD
	return sstats
}

func GammaLn2D(param [][]float64) [][]float64 {
	result := make([][]float64, len(param))

	for i := 0; i < len(param); i++ {
		innerLen := len(param[i])
		row := make([]float64, innerLen)
		for j := 0; j < innerLen; j++ {
			gln := C.gamma(C.double(row[i]))
			row = append(row, float64(gln))
		}
		result = append(result, row)
	}
	return result
}

func GammaLn(param []float64) []float64 {
	result := make([]float64, len(param))

	for i := 0; i < len(param); i++ {
		gln := C.gamma(C.double(param[i]))
		result = append(result, float64(gln))
	}
	return result
}

func GammaLnScalar(param float64) float64 {
	gln := float64(C.gamma(C.double(param)))
	return gln
}


func Exp1D(values []float64) []float64 {
	result := make([]float64, len(values))
	for i := 0; i < len(values); i++ {
		result[i] = math.Exp(values[i])
	}
	return result
}

func Exp(values [][]float64) [][]float64 {
	/*exps := make([]float64, 0)
	for i := 0; i < len(samples); i++ {
		exp := samples[i]*samples[i]
		exps = append(exps, exp)
	}*/
	for i := 0; i < len(values); i++ {
		value := values[i]
		for j := 0; j < len(value); j++ {
			values[i][j] = math.Exp(values[i][j])
		}
	}

	return values
}

func DirichletExpectation(sstats []float64) []float64 {
	/*
	np.sum([[0, 1], [0, 5]], axis=0)
    array([0, 6])

	[[0, 1],
	 [0, 5]]

	 [0, 6]

	>>> np.sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])

    [[0, 1],   [1,
     [0, 5]]    5]

	 */
	alpha := make([]float64, len(sstats))
	var sum float64

	//result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
	for i := 0; i < len(sstats); i++ {
		sum = sum + sstats[i]
	}
	sumOfPsi := C.psi(C.double(sum))

	for i := 0; i < len(sstats); i++ {
		temp := C.psi(C.double(sstats[i])) - sumOfPsi
		alpha = append(alpha, float64(temp))
	}
	return alpha
}

func PsiScalar(A float64) float64 {
	result := float64(C.psi(C.double(A)))

	return result
}

func PsiVector(A []float64) []float64 {
	result := make([]float64, len(A))
	for i := 0; i < len(A); i++ {
		temp := float64(C.psi(C.double(A[i])))
		result = append(result, temp)
	}
	return result
}

func DirichletExpectation2D(alpha [][]float64) [][]float64 {
	/*
	"""
    For a vector `theta~Dir(alpha)`, compute `E[log(theta)]`.

    """
    if (len(alpha.shape) == 1):
        result = psi(alpha) - psi(np.sum(alpha))
    else:
        result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
    return result.astype(alpha.dtype)  # keep the same precision as input
	 */

	/*
	np.sum([[0, 1], [0, 5]], axis=0)
    array([0, 6])

	[[0, 1],
	 [0, 5]]

	 [0, 6]

	>>> np.sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])

    [[0, 1],   [1,
     [0, 5]]    5]

	 */

	fmt.Println("len(alpha)", len(alpha))
	var sumOfRow []float64 = make([]float64, 0)

	sum := 0.0
	//result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
	for i := 0; i < len(alpha); i++ {
		innerlen := len(alpha[i])
		for j := 0; j < innerlen; j++ {
			sum += alpha[i][j]
		}
		temp := C.psi(C.double(sum))
		sumOfRow = append(sumOfRow, float64(temp))
		sum = 0
	}
	/*
		twoD := make([][]int, shape[0])
		newSize := shape[0] * shape[1]

		for i := 0; i < newSize; i++ {
			innerLen := shape[1]
			twoD[i] = make([]int, innerLen)
			for j := 0; j < innerLen; j++ {
				twoD[i][j] = Gamma{shape, scale, src}.Rand()
			}
		}
		return twoD
	 */

	fmt.Println("len(sumOfRow)", len(sumOfRow))
	var sumOfPsi [][]float64 = make([][]float64, len(sumOfRow))
	for i := 0; i < len(sumOfRow); i++ {
		sumOfPsi[i] = make([]float64, 1)
		for j := 0; j < 1; j++ {
			sumOfPsi[i][j] = sumOfRow[i]
		}
	}

	result := make([][]float64, len(alpha))
	fmt.Println("len(alpha)", len(alpha))
	for i := 0; i < len(alpha); i++ {
		innerLen := len(alpha[i])
		result[i] = make([]float64, innerLen)
		for j := 0; j < innerLen; j++ {
			temp := float64(C.psi(C.double(alpha[i][j]))) - sumOfPsi[i][0]
			result[i][j] = float64(temp)
		}
	}
	return result
}

func min(a int, b int) int {
	if a < b {
		return a
	} else {
		return b
	}
}

func max(a int, b int) int {
	if a > b {
		return a
	} else {
		return b
	}
}

// Max returns the maximum value in the input slice. If the slice is empty, Max will panic.
func Max(s [][]float64) float64 {
	x := 0.0
	for i:= 0; i < len(s); i++ {
		x = s[i][MaxIdx(s[i])]
	}
	return x
}
// MaxIdx returns the index of the maximum value in the input slice. If several
// entries have the maximum value, the first such index is returned. If the slice
// is empty, MaxIdx will panic.
func MaxIdx(s []float64) int {
	if len(s) == 0 {
		panic("floats: zero slice length")
	}
	max := s[0]
	var ind int
	for i, v := range s {
		if v > max {
			max = v
			ind = i
		}
	}
	return ind
}

// LogSumExp returns the log of the sum of the exponentials of the values in s.
// Panics if s is an empty slice.
func LogSumExp(s [][]float64) float64 {
	// Want to do this in a numerically stable way which avoids
	// overflow and underflow
	// First, find the maximum value in the slice.
	maxval := Max(s)
	if math.IsInf(maxval, 0) {
		// If it's infinity either way, the logsumexp will be infinity as well
		// returning now avoids NaNs
		return maxval
	}
	var lse float64
	// Compute the sumexp part
	for _, row := range s {
		for _, val := range row {
			lse += math.Exp(val - maxval)
		}
	}
	// Take the log and add back on the constant taken out
	return math.Log(lse) + maxval
}

func zerosLike(shape []int) [][]float64 {
	twoD := make([][]float64, shape[0])

	for i := 0; i < shape[0]; i++ {
		innerLen := shape[1]
		twoD[i] = make([]float64, innerLen)
		for j := 0; j < innerLen; j++ {
			twoD[i] = append(twoD[i], 0.0)
		}
	}
	return twoD
}

func getMatrixRows(matrix [][]float64, docId int) ([]float64, error) {
	result := make([]float64, 0)
	if matrix != nil {
		for ctr, value := range matrix {
			if ctr == docId {
				for _, val := range value {
					result = append(result, val)
				}
			}
		}
	} else {
		return result, errors.New("Nothing to process")
	}
	return result, nil
}

func Reshape(value []float64, shape int) []float64 {
	return value
}

//func AddToMatrixRowsForIds(docId []int, matrix [][]float64, value [][]float64) error {
//	if matrix != nil || value != nil {
//		for ctr, value := range matrix {
//			if ctr == docId {
//				for idx, val := range value {
//					matrix [ctr][idx] = val
//				}
//			}
//		}
//	} else {
//		return errors.New("Nothing to process")
//	}
//	return nil
//}

func SetMatrixFromVector(doctId int, M [][]float64, v[]float64) {
	for idx, value := range v {
		M[doctId][idx] = value
	}
}

func SetSStasRowsForDocIds(docIds []int, matrix [][]float64, values [][]float64) error {
	if matrix != nil || values != nil || docIds != nil {
		/*
			ids
			[0, 1, 2, 3]
			sstats
			[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
			 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]
			numpy.outer(expElogthetad.T, cts / phinorm)
			[[  7.7131958   49.19957986  17.34944568   6.1855617 ]
			 [  0.60159045   3.83731961   1.35316944   0.48244268]]
			ids
			[0, 2, 4, 5, 6]
			sstats
			[[  7.7131958   49.19957986  17.34944568   6.1855617    0.           0.
				0.           0.           0.           0.           0.           0.
				0.           0.           0.           0.        ]
			 [  0.60159045   3.83731961   1.35316944   0.48244268   0.           0.
				0.           0.           0.           0.           0.           0.
				0.           0.           0.           0.        ]]
		 */
		for i := 0; i < len(docIds); i++ {
			for idx, value := range values {
				temp := matrix[idx][docIds[i]]
				temp = temp + value[i]
				matrix[idx][docIds[i]] = temp
			}
		}

	} else {
		return errors.New("Nothing to process")
	}
	return nil
}

func getMatrixColumnsById(matrix [][]float64, docId int) []float64 {
	result := make([]float64, len(matrix))
	if matrix != nil {
		for ctr, value := range matrix {
			for i := 0; i < len(value); i++ {
				if ctr == i {
					result = append(result, value[i])
				}
			}
		}
	} else {
		//return result, errors.New("Nothing to process")
		fmt.Print("Nothing to process")
	}
	return result
}

func getMatrixColumnsByIds(matrix [][]float64, docIds []int) ([][]float64, error) {
	result := make([][]float64, len(matrix))
	if matrix != nil && docIds != nil {
		for ctr, value := range matrix {
			result[ctr] = make([]float64, len(docIds))
			for i := 0; i < len(docIds); i++ {
				if ctr == i {
					result = append(result, value)
				}
			}
		}
	} else {
		return result, errors.New("Nothing to process")
	}
	return result, nil
}

func DotVM(A []float64, B [][]float64) []float64 {

	//  A = [  9.80750921  56.75183971  21.41869303   7.11344183]
	/* B =
		[[ 0.12605954  0.04600815]
		[ 0.01862523  0.02179819]
		[ 0.11019805  0.06512481]
		[ 0.14784765  0.17718007]]
	 */
	/*
	calculation : 9.80750921×0.12605954+56.75183971×0.01862523+21.41869303×0.11019805+7.11344183×0.14784765
	 */
	colsA := len(A)
	rowsB := len(B)
	colsB := func() int {
		colsB := 0
		for i := 0; i < rowsB; i++ {
			colsB = len(B[i])
			break
		}
		return colsB
	}()

	outer := make([][]float64, colsB)
	result := make([]float64, colsB)
	if colsA != rowsB {
		fmt.Println("Matrix A and B col and row mismatch")
	} else {
		inner := make([]float64, colsB)
		for i := 0; i < len(A); i++ {
			for j := 0; j < colsB; j++ {
				inner = append(inner, A[i] * B[i][j])
				//[a11, a12, a13]
				//[a21, a22, a23]
			}
			outer = append(outer, inner)
		}
	}

	for i := 0; i < len(outer); i++ {
		row := outer[i]
		for j := 0; j < len(row); j++ {
			elem := row[j]
			result[j] += elem
		}
	}
	return result
}

func CopyVector(A []float64) []float64{
	result := make([]float64, len(A))

	for i := 0; i <= len(A); i++ {
		result = append(result, A[i])

	}
	return result
}

//func DotMM(A [][]float64, B [][]float64) ([][]float64, error) {
//	colsA := len(A)
//	rowsA := func() int {
//		rows:=0
//		for i:=0; i < len(A); i++ {
//			rows = i
//		}
//		return rows + 1
//	}()
//	colsB := len(B)
//	rowsB := func() int {
//		rows:=0
//		for i:=0; i < len(B); i++ {
//			rows = i
//		}
//		return rows + 1
//	}()
//	var mult float64
//	var multSum float64
//	result := make([][]float64, rowsA)
//	if colsA != rowsB {
//		return result, errors.New("Matrix A and B col and row mismatch")
//	} else {
//
//		for _, value := range A {
//			for idx, value1 := range B {
//				result[idx] = make([]float64, colsB)
//				mult = value * value1
//				multSum = multSum + mult
//				result = append(result, multSum)
//			}
//		}
//	}
//	return result, nil
//}

func transposeV(A []float64) []float64 {
	colsA := len(A)

	transpose := make([]float64, colsA)
	for m:=0; m < colsA; m++ {
		transpose = append(transpose, A[m])
	}
	return transpose
}

func transposeM(A [][]float64) [][]float64 {
	colsA := len(A)
	rowsA := func() int {
		rows:=0
		for i:=0; i < len(A); i++ {
			rows = i
		}
		return rows + 1
	}()
	transpose := make([][]float64, colsA)
	for m:=0; m < len(A); m++ {
		row := A[m]
		transpose[m] = make([]float64, rowsA)
		for n:=0; n < len(row); n++ {
			temp := row[n]
			transpose[m] = append(transpose[m], temp)
		}
	}
	return transpose
}

func Outer(a []float64, b []float64) [][]float64 {
	lenOfA := len(a)
	lenOfB := len(b)
	result := make([][]float64, min(lenOfA, lenOfB))

	for i := 0; i < len(a); i++ {
		innerLen := max(lenOfA, lenOfB)
		result[i] = make([]float64, innerLen)
		for j := 0; j < len(b); j++ {
			temp := a[i] * b[j]
			result[i] = append(result[i], temp)
		}
	}
	return result
}

func Multiply(a [][]float64, b [][]float64)  [][]float64 {
	result := make([][]float64, len(a))
	for idx, value := range a {
		result[idx] = make([]float64, len(value))
		for i := 0; i < len(value); i++ {
			result[idx][i] = a[idx][i] * b[idx][i]
		}
		/*for idx1, value1 := range b {
			result[idx][idx1] = value[idx1] * value1
		}*/
	}
	return result
}

func MultiplyV(a []float64, b []float64)  []float64 {
	result := make([]float64, len(a))
	if len(a) == len(b) {
		for i:= 0; i < len(a); i++ {
			result[i] = a[i] * b[i]
		}
	}
	return result
}

func MultMS(M [][]float64, S float64) [][]float64 {
	result := make([][]float64, len(M))
	for idx, row := range M {
		result[idx] = make([]float64, len(row))
		for _, col := range row {
			result[idx] = append(result[idx], col)
		}
	}
	return result
}

func MultSV(S float64, V []float64) []float64 {
	result := make([]float64, len(V))
	for i := 0; i < len(V); i++ {
		V[i] = S * V[i]
	}
	return result
}

func MultiplyVectorScalar(V []float64, S float64) []float64 {
	result := make([]float64, len(V))
	for i := 0; i < len(V); i++ {
		result = append(result, S * V[i])
	}
	return result
}

func AddV(a []float64, b []float64) []float64 {
	result := make([]float64, len(a))
	if len(a) == len(b) {
		for i:= 0; i < len(a); i++ {
			result[i] = a[i] + b[i]
		}
	}
	return result
}

func AddMV(a [][]float64, b []float64) [][]float64 {
	result := make([][]float64, len(a))

	for i:= 0; i < len(a); i++ {
		row := a[i]
		result[i] = make([]float64, len(row))
		for j:=0; j < len(row); j++ {
			result[i] = append(result[i], a[i][j] + b[i])
		}
	}
	return result
}

func AddMM(A [][]float64, B[][]float64) [][]float64 {
	result := make([][]float64, len(A))

	for i:= 0; i < len(A); i++ {
		row := A[i]
		result[i] = make([]float64, len(row))
		for j:=0; j < len(row); j++ {
			result[i] = append(result[i], A[i][j] + B[i][j])
		}
	}
	return result
}

func DivideV(V1 []int, V2 []float64) []float64 {
	result := make([]float64, len(V1))
	if len(V1) == len(V2) {
		for i := 0; i < len(V1); i++ {
			result = append(result, float64(V1[i]) / V2[i])
		}
	}
	return result
}
func DivideVByScalar(V []float64, S float64) []float64 {
	result := make([]float64, 0)
	for i := 0; i < len(V); i++ {
		result = append(result, V[i] / S)
	}
	return result
}
func AddVerySmallNumber(values []float64) []float64 {
	len := len(values)
	result := make([]float64, len)
	for i := 0; i < len; i++ {
		result[i] = values[i] + 1e-100
	}
	return result
}

func Sum(values []float64) float64 {
	len := len(values)
	result := 0.0
	for i := 0; i < len; i++ {
		result = result + values[i]
	}
	return result
}

func SumM(values [][]float64) float64 {
	result := 0.0
	for i := 0; i < len(values); i++ {
		row := values[i]
		for j := 0; j < len(row); j++ {
			result = result + values[i][j]
		}
	}
	return result
}

func SumAxis1(values [][]float64) []float64 {
	result := make([]float64, 2)
	for i := 0; i < len(values); i++ {
		sum := 0.0
		innerLen:= len(values[i])
		for j := 0; j < innerLen; j++ {
			sum = sum + values[i][j]
		}
		result = append(result, sum)
	}
	return result
}


//--------------------------------------------------------------------------------------------------
// The main function
//--------------------------------------------------------------------------------------------------

func main() {
	//sstats := Gamma2D(100., 1. / 100., []int{2, 16})
	//fmt.Println(sstats)

	/*tstst := [][]float64{{1.04897752, 0.74388644, 1.05618059, 1.02939178, 1.08430247,0.95599866, 1.22568971, 0.92028319, 0.83144373, 1.06974302, 0.85080179, 0.95261723, 0.98513228, 0.97925246, 0.86092491, 1.18379171},
		{0.89808978, 1.07810982, 0.99313994, 1.22257276, 1.03377996, 1.12488726, 0.95504473, 1.00532363, 0.79771307, 0.9160643, 1.0424362, 1.11101386, 1.18292857, 0.99062379, 0.92100996, 1.11226653}}

	output := DirichletExpectation2D(tstst)
	fmt.Println(output)
	fmt.Println(Exp(output))*/

	//fmt.Println(math.Exp(-3.22603248))

	texts := make([][]string, 0)
	text := make([]string, 0)

	text = []string{"bank", "river", "shore", "water", "river"}
	texts = append(texts, text)

	text = []string{"river", "water", "flow", "fast", "tree"}
	texts = append(texts, text)

	text = []string{"bank", "water", "fall", "flow"}
	texts = append(texts, text)

	text = []string{"bank", "bank", "water", "rain", "river"}
	texts = append(texts, text)

	text = []string{"river", "water", "mud", "tree"}
	texts = append(texts, text)

	text = []string{"money", "transaction", "bank", "finance"}
	texts = append(texts, text)

	text = []string{"bank", "borrow", "money"}
	texts = append(texts, text)

	text = []string{"bank", "finance"}
	texts = append(texts, text)

	text = []string{"finance", "money", "sell", "bank"}
	texts = append(texts, text)

	text = []string{"borrow", "sell"}
	texts = append(texts, text)

	text = []string{"bank", "loan", "sell"}
	texts = append(texts, text)

	dictionary := NewDictionary(texts)
	//corpus := make([][]int, 0)
	corpus := make([]map[int]int, 0)

	for _, text := range texts {
		result, _ := dictionary.doc2bow(text, true, false)
		corpus = append(corpus, result)
	}

	fmt.Println("corpus", len(corpus))
	ldaParams := NewLDAParams()
	ldaParams.Corpus = corpus
	ldaParams.Id2Word = dictionary
	ldaParams. NumTopics = 2

	model := NewLDAModel(ldaParams)
	if buildErr := model.BuildModel(ldaParams); buildErr != nil {
		fmt.Println("buildErr", buildErr)
	}
	populateErr := model.PopulateAlphaAndETA()
	fmt.Println("populateErr", populateErr)
	fmt.Println(model.ExpElogbeta)

}
